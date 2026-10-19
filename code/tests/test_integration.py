"""Regressions discovered in the official comparison, plus integrated workflows.
No external model requests; scripted responses are labelled fixtures.
"""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from tracecoder.catalog import DATA, get_demo
from tracecoder.engine import Config
from tracecoder.providers import FixtureProvider, build_messages, parse_response
from tracecoder.runner import Runner
from tracecoder.service import Workbench
from tracecoder.storage import validate_task
from web_demo import create_app


class OfficialContractTests(unittest.TestCase):
    def test_plain_asserts_ignore_spacing_and_continue_after_failure(self):
        result = Runner().evaluate('def double(x): return x\n', {'test': 'assert double(0)==0\nassert double(2)==4\nassert double(3) == 6', 'entry_point': ''})
        self.assertTrue(result['valid'])
        self.assertFalse(result['passed'])
        self.assertEqual((result['passed_count'], result['total_count']), (1, 3))

    def test_asserts_preserve_setup_and_loop_inputs(self):
        result = Runner().evaluate('def same(x): return x\n', {'test': 'values=[1,2,3]\nfor x in values:\n    assert same(x)==x\n', 'entry_point': ''})
        self.assertTrue(result['passed'])
        self.assertEqual(result['total_count'], 3)

    def test_mixed_test_entrypoints_cannot_silently_omit_failures(self):
        tests = 'assert True\nimport unittest\nclass T(unittest.TestCase):\n    def test_fail(self): self.assertTrue(False)\n'
        result = Runner().evaluate('x=1', {'test': tests, 'entry_point': ''})
        self.assertFalse(result['valid'])
        self.assertFalse(result['passed'])
        with self.assertRaises(ValueError):
            validate_task({'title': 'mixed', 'description': 'test', 'initial_code': 'x=1', 'test': tests})

    def test_timeout_unittest_is_never_one_of_one_pass(self):
        result = Runner(timeout=1).evaluate('def hang():\n    while True: pass\n', {'test': 'import unittest\nclass T(unittest.TestCase):\n    def test_hang(self): hang()\n', 'entry_point': ''})
        self.assertFalse(result['passed'])
        self.assertFalse(result['valid'])
        self.assertEqual(result['kind'], 'timeout')

    def test_full_history_not_only_last_two_plans(self):
        history = [{'round': i, 'repair_plan': ['plan-%d' % i], 'repaired_code': 'candidate-%d' % i, 'error_feedback': 'feedback-%d' % i, 'passed_count': i} for i in range(5)]
        prompt = build_messages('analysis', {'description': 'task', 'instrumented_code': 'code', 'runtime_trace': 'trace', 'lesson_record': history})[1]['content']
        for i in range(5):
            for field in ('plan', 'candidate', 'feedback'):
                self.assertIn('%s-%d' % (field, i), prompt)

    def test_official_code_and_analysis_formats(self):
        raw = 'REPAIR_PLAN_START\n```python\nUse strict lower bound.\n```\nREPAIR_PLAN_END\nInstrumentation_Suggestions_START\nInspect start and timestamp.\nInstrumentation_Suggestions_END'
        self.assertEqual(parse_response('analysis', raw)['repair_plan'], ['Use strict lower bound.'])
        raw = 'Repaired CODE Part:\n```python\nfragment\n```\nCODE:\n```python\ndef f(): return 1\n```\nEND_CODE'
        self.assertEqual(parse_response('repair', raw)['code'], 'def f(): return 1')
        self.assertEqual(parse_response('generation', '```python\ndef f(): return 1\n```')['code'], 'def f(): return 1')
        with self.assertRaises(ValueError):
            parse_response('repair', '```python\ndef f(): return 1\n```')

    def test_config_rejects_booleans_as_numeric_budget(self):
        for kwargs in ({'timeout': True}, {'max_rounds': 1.5}, {'enable_instrumentation': 'false'}):
            with self.assertRaises(ValueError):
                Config(**kwargs)


class OfficialWebIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workbench = Workbench(state_dir=self.temp.name)
        self.addCleanup(self.workbench.close)
        self.app = create_app(self.workbench, token='')
        self.client = self.app.test_client()

    def wait(self, run):
        deadline = time.monotonic() + 10
        while run['status'] in ('queued', 'running') and time.monotonic() < deadline:
            time.sleep(.02)
            run = self.workbench.get(run['id'])
        self.assertEqual(run['status'], 'completed', run.get('error'))
        return run

    def test_official_pages_and_new_system_pages_render(self):
        for route in ('/', '/generate', '/history', '/examples', '/settings', '/help', '/workbench', '/projects', '/deployment'):
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200, route)
            self.assertIn(b'data-initial-view=', response.data)
            self.assertNotIn(b'cdn.', response.data)
            self.assertIn("script-src 'self'", response.headers['Content-Security-Policy'])

    def test_original_examples_preserved_and_statically_importable(self):
        expected = json.loads((DATA / 'official/examples.json').read_text())
        self.assertEqual(self.client.get('/api/examples').json, expected)
        self.assertEqual(len(expected), 12)
        for ex in expected:
            task = validate_task({'title': ex['name'], 'description': ex['description'], 'test': ex['test_case'], 'start_mode': 'generate'})
            self.assertEqual(task['test'], ex['test_case'])
        self.assertEqual(self.client.get('/api/examples/example6').json, expected[5])

    def test_official_parameters_change_actual_execution(self):
        response = self.client.post('/api/run_tracecoder', json={'mode': 'fixture', 'task_id': 'paper-sunday', 'max_debug_attempts': 1, 'max_no_improvement': 4, 'timeout': 5, 'enable_instrumentation': False})
        self.assertEqual(response.status_code, 202)
        run = self.wait(response.json)
        self.assertEqual(run['config'], {'max_rounds': 1, 'patience': 4, 'timeout': 5, 'enable_instrumentation': False})
        self.assertEqual(run['stop_reason'], 'round_budget_exhausted')
        self.assertFalse(run['final_eval']['passed'])
        self.assertEqual([c['role'] for c in run['calls']], ['analysis', 'repair'])
        self.assertTrue(any(e['stage'] == 'instrumentation_disabled' for e in run['events']))

    def test_stream_result_is_same_saved_run_and_real_test_result(self):
        response = self.client.post('/api/run_tracecoder_stream', json={'mode': 'fixture', 'task_id': 'api-window'})
        self.assertEqual(response.status_code, 200)
        text = response.get_data(as_text=True)
        blocks = [block for block in text.split('\n\n') if block.startswith('event: result\n')]
        self.assertEqual(len(blocks), 1)
        run = json.loads(blocks[0].split('data: ', 1)[1])
        self.assertEqual(run, self.client.get('/api/runs/' + run['id']).json)
        self.assertTrue(run['final_eval']['passed'])
        self.assertEqual(run['evidence'], '预设智能体响应 + 本次真实代码执行；不属于模型实验')

    def test_generation_versions_saved_without_model_or_execution(self):
        task = {'title': 'Generation input', 'description': 'Return double(x)', 'test': 'assert double(3)==6', 'start_mode': 'generate'}
        first = self.client.post('/api/generation/tasks', json={'task': task})
        self.assertEqual(first.status_code, 201)
        revision = first.json
        second = self.client.post('/api/generation/tasks', json={'task': {**task, 'test': 'assert double(0)==0'}, 'task_id': revision['id'], 'project_id': revision['project_id']})
        self.assertEqual(second.json['revision'], 2)
        self.assertEqual(self.workbench.store.task(revision['id'], 1)['test'], task['test'])
        self.assertEqual(self.workbench.history(), [])
        response = self.client.post('/api/run_tracecoder', json={'task_id': revision['id'], 'revision': 1})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(self.workbench.history(), [])

    def test_generation_and_repair_share_engine_without_passing_tests_to_generation(self):
        fixture = get_demo('api-window')
        task = {'title': 'Scripted generation acceptance', 'description': fixture['description'], 'test': fixture['test'], 'start_mode': 'generate'}
        saved = self.client.post('/api/generation/tasks', json={'task': task}).json
        class ScriptedGeneration(FixtureProvider):
            def complete(self, role, context, number):
                if role == 'generation':
                    return {'code': fixture['initial_code']}, {'raw_response': fixture['initial_code'], 'response_source': 'authored_fixture_not_llm'}
                return super().complete(role, context, number)
        with patch.object(self.workbench, 'config', return_value={'live_enabled': True}), patch('tracecoder.service.CompatibleProvider', return_value=ScriptedGeneration(fixture)), patch('tracecoder.service.Runner', return_value=Runner()):
            response = self.client.post('/api/run_tracecoder', json={'task_id': saved['id'], 'revision': 1})
            self.assertEqual(response.status_code, 202, response.json)
            run = self.wait(response.json)
        self.assertEqual(run['start_mode'], 'generate')
        self.assertTrue(run['final_eval']['passed'])
        self.assertEqual(run['task']['revision'], 1)
        self.assertNotIn(fixture['test'], run['calls'][0]['messages'][1]['content'])
        self.assertEqual(run['calls'][0]['role'], 'generation')

    def test_keys_and_unversioned_overrides_rejected(self):
        for body in ({'api_key': 'do-not-store'}, {'base_url': 'https://example.invalid'}, {'model_name': 'override'}, {'mode': 'fixture', 'test_cases': 'assert True'}, {'task_id': 'abc', 'test_cases': 'assert True'}):
            self.assertEqual(self.client.post('/api/run_tracecoder', json=body).status_code, 400)
        self.assertEqual(self.client.post('/api/stop_tracecoder', json={}).status_code, 400)
        result = self.client.post('/api/validate_config', json={}).json
        self.assertFalse(result['model_called'])
        self.assertEqual(self.client.post('/api/validate_config', json={'api_key': 'do-not-store'}).status_code, 400)

    def test_public_demo_prevents_new_generation_input_storage(self):
        self.workbench.public_demo = True
        response = self.client.post('/api/generation/tasks', json={'task': {'title': 'x'}})
        self.assertEqual(response.status_code, 403)
        self.assertEqual(self.workbench.store.projects(), [])


if __name__ == '__main__':
    unittest.main()
