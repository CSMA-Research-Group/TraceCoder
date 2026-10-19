#!/usr/bin/env python3
"""Integrated official Flask WebDemo, served by Waitress (no debug server)."""
import argparse
import os
from pathlib import Path
import signal

from tracecoder.catalog import DATA
from tracecoder.service import Workbench  # Compatibility for integrations/tests.


def main():
    parser = argparse.ArgumentParser(description="TraceCoder integrated official WebDemo")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--public-demo", action="store_true", help="Immutable fixtures only; no private project execution")
    args = parser.parse_args()
    token = os.environ.get("TRACECODER_ACCESS_TOKEN", "")
    local_only = args.host in ("127.0.0.1", "localhost", "::1")
    if not local_only and not token and not args.public_demo:
        parser.error("非本机监听需要 TRACECODER_ACCESS_TOKEN，或显式指定 --public-demo。")
    import fcntl
    from waitress import create_server
    from web_demo import create_app
    state_dir = Path(os.environ.get("TRACECODER_STATE_DIR", str(DATA)))
    (state_dir / "workspace").mkdir(parents=True, exist_ok=True)
    lock_file = (state_dir / "workspace/.server.lock").open("a")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error("该数据目录已有服务运行；请使用现有实例或设置不同的 TRACECODER_STATE_DIR。")
    workbench = Workbench(args.public_demo, state_dir)
    server = create_server(create_app(workbench, token, local_only), host=args.host, port=args.port,
                           threads=12, max_request_body_size=250000, channel_timeout=60,
                           clear_untrusted_proxy_headers=True, ident="TraceCoder")
    def stop_service(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop_service)
    print("TraceCoder ready: http://%s:%d (Flask / Waitress)" % (args.host, args.port), flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        workbench.close()
        server.close()
        lock_file.close()


if __name__ == "__main__":
    main()
