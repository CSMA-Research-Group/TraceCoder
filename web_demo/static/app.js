function showToast(type, title, message) {
    const toast = document.getElementById('notificationToast');
    const toastIcon = document.getElementById('toastIcon');
    const toastTitle = document.getElementById('toastTitle');
    const toastMessage = document.getElementById('toastMessage');
    
    const icons = {
        'success': 'fas fa-check-circle text-success',
        'error': 'fas fa-times-circle text-danger',
        'warning': 'fas fa-exclamation-triangle text-warning',
        'info': 'fas fa-info-circle text-info'
    };
    
    toastIcon.className = icons[type] || icons['info'];
    toastTitle.textContent = title;
    toastMessage.textContent = message;
    
    const bsToast = new bootstrap.Toast(toast);
    bsToast.show();
}

function escapeHtml(text) {
    if (!text) return '';
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    if (!timestamp) return '-';
    const date = new Date(timestamp);
    const now = new Date();
    const diff = now - date;
    
    if (diff < 60000) return '刚刚';
    if (diff < 3600000) return Math.floor(diff / 60000) + '分钟前';
    if (diff < 86400000) return Math.floor(diff / 3600000) + '小时前';
    if (diff < 604800000) return Math.floor(diff / 86400000) + '天前';
    
    return date.toLocaleDateString('zh-CN');
}

function formatDuration(seconds) {
    if (!seconds) return '0s';
    seconds = parseFloat(seconds);
    
    if (seconds < 60) {
        return seconds.toFixed(1) + 's';
    } else if (seconds < 3600) {
        const minutes = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${minutes}m ${secs}s`;
    } else {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m`;
    }
}

function copyToClipboard(text) {
    if (navigator.clipboard && navigator.clipboard.writeText) {
        return navigator.clipboard.writeText(text)
            .then(() => {
                showToast('success', '已复制', '内容已复制到剪贴板');
                return true;
            })
            .catch(() => {
                showToast('error', '复制失败', '无法复制到剪贴板');
                return false;
            });
    }
    
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.style.position = 'fixed';
    textarea.style.opacity = '0';
    document.body.appendChild(textarea);
    textarea.select();
    
    try {
        document.execCommand('copy');
        showToast('success', '已复制', '内容已复制到剪贴板');
        return Promise.resolve(true);
    } catch (err) {
        showToast('error', '复制失败', '无法复制到剪贴板');
        return Promise.resolve(false);
    } finally {
        document.body.removeChild(textarea);
    }
}

function downloadFile(content, filename, mimeType) {
    mimeType = mimeType || 'text/plain';
    
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.style.display = 'none';
    
    document.body.appendChild(a);
    a.click();
    
    setTimeout(() => {
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }, 100);
}

function debounce(func, wait) {
    let timeout;
    return function executedFunction(...args) {
        const later = () => {
            clearTimeout(timeout);
            func(...args);
        };
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
    };
}

function throttle(func, limit) {
    let inThrottle;
    return function(...args) {
        if (!inThrottle) {
            func.apply(this, args);
            inThrottle = true;
            setTimeout(() => inThrottle = false, limit);
        }
    };
}

function initTheme() {
    const config = JSON.parse(localStorage.getItem('tracecoder_config') || '{}');
    const theme = config.theme || 'light';
    
    if (theme === 'dark') {
        document.body.classList.add('dark-theme');
    } else if (theme === 'auto') {
        if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
            document.body.classList.add('dark-theme');
        }
    }
    
    const themeToggle = document.getElementById('themeToggle');
    if (themeToggle) {
        themeToggle.checked = theme === 'dark';
        themeToggle.addEventListener('change', function() {
            const newTheme = this.checked ? 'dark' : 'light';
            document.body.classList.toggle('dark-theme', this.checked);
            
            const config = JSON.parse(localStorage.getItem('tracecoder_config') || '{}');
            config.theme = newTheme;
            localStorage.setItem('tracecoder_config', JSON.stringify(config));
        });
    }
}

function checkApiStatus() {
    const config = JSON.parse(localStorage.getItem('tracecoder_config') || '{}');
    const statusBadge = document.getElementById('apiStatus');
    
    if (statusBadge) {
        if (config.apiKey) {
            statusBadge.innerHTML = '<i class="fas fa-circle-dot me-2" style="font-size: 8px;"></i> 已配置';
            statusBadge.classList.remove('bg-secondary', 'bg-success', 'bg-danger');
            statusBadge.classList.add('bg-success');
        } else {
            statusBadge.innerHTML = '<i class="fas fa-circle-dot me-2" style="font-size: 8px;"></i> 未配置';
            statusBadge.classList.remove('bg-secondary', 'bg-success', 'bg-danger');
            statusBadge.classList.add('bg-secondary');
        }
    }
}

function initHighlight() {
    if (typeof hljs !== 'undefined') {
        hljs.configure({
            languages: ['python']
        });
        hljs.highlightAll();
    }
}

function formatPythonCode(code) {
    if (!code) return code;
    
    let lines = code.split('\n');
    let formatted = [];
    let indentLevel = 0;
    const indentStr = '    ';
    
    lines.forEach(line => {
        let trimmed = line.trim();
        
        if (!trimmed) {
            formatted.push('');
            return;
        }
        
        if (trimmed.endsWith(':')) {
            formatted.push(indentStr.repeat(indentLevel) + trimmed);
            indentLevel++;
        } else if (trimmed.startsWith('return ') || trimmed.startsWith('break') || trimmed.startsWith('continue') || trimmed.startsWith('pass')) {
            if (indentLevel > 0 && formatted[formatted.length - 1] === '') {
                indentLevel = Math.max(0, indentLevel - 1);
            }
            formatted.push(indentStr.repeat(indentLevel) + trimmed);
        } else if (trimmed === 'else:' || trimmed === 'elif ' || trimmed === 'except:' || trimmed === 'finally:') {
            indentLevel = Math.max(0, indentLevel - 1);
            formatted.push(indentStr.repeat(indentLevel) + trimmed);
            indentLevel++;
        } else {
            formatted.push(indentStr.repeat(indentLevel) + trimmed);
        }
    });
    
    return formatted.join('\n');
}

function initKeyboardShortcuts() {
    document.addEventListener('keydown', function(e) {
        if (e.ctrlKey || e.metaKey) {
            switch(e.key.toLowerCase()) {
                case 'enter':
                    e.preventDefault();
                    const runButton = document.getElementById('runButton');
                    if (runButton && runButton.style.display !== 'none') {
                        runButton.click();
                    }
                    break;
                case 's':
                    e.preventDefault();
                    showToast('info', '提示', '配置已自动保存到本地');
                    break;
                case 'b':
                    e.preventDefault();
                    const themeToggle = document.getElementById('themeToggle');
                    if (themeToggle) {
                        themeToggle.click();
                    }
                    break;
            }
        }
        
        if (e.key === 'Escape') {
            const modals = document.querySelectorAll('.modal.show');
            modals.forEach(modal => {
                const bsModal = bootstrap.Modal.getInstance(modal);
                if (bsModal) {
                    bsModal.hide();
                }
            });
        }
    });
}

function initCodeEditorEnhancements() {
    const textareas = document.querySelectorAll('.code-editor, textarea[id="testCases"], textarea[id="problemDescription"]');
    
    textareas.forEach(textarea => {
        textarea.addEventListener('keydown', function(e) {
            if (e.key === 'Tab') {
                e.preventDefault();
                const start = this.selectionStart;
                const end = this.selectionEnd;
                
                this.value = this.value.substring(0, start) + '    ' + this.value.substring(end);
                this.selectionStart = this.selectionEnd = start + 4;
            }
            
            if (e.key === 'Enter') {
                const start = this.selectionStart;
                const lineStart = this.value.lastIndexOf('\n', start - 1) + 1;
                const currentLine = this.value.substring(lineStart, start);
                const indent = currentLine.match(/^(\s*)/)[1];
                
                if (currentLine.trimEnd().endsWith(':')) {
                    e.preventDefault();
                    const newIndent = indent + '    ';
                    const pos = this.selectionStart;
                    this.value = this.value.substring(0, pos) + '\n' + newIndent + this.value.substring(pos);
                    this.selectionStart = this.selectionEnd = pos + 1 + newIndent.length;
                }
            }
        });
        
        textarea.addEventListener('input', function() {
            this.style.height = 'auto';
            this.style.height = Math.min(this.scrollHeight, 400) + 'px';
        });
    });
}

function animateNumber(element, target, duration) {
    const start = parseInt(element.textContent) || 0;
    const increment = (target - start) / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= target) || (increment < 0 && current <= target)) {
            element.textContent = target;
            clearInterval(timer);
        } else {
            element.textContent = Math.round(current);
        }
    }, 16);
}

function showLoading(container, message) {
    message = message || '加载中...';
    container.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
            <p class="text-muted mt-3">${message}</p>
        </div>
    `;
}

function showError(container, message, retryCallback) {
    let html = `
        <div class="text-center py-5">
            <i class="fas fa-exclamation-triangle fa-3x text-danger mb-3"></i>
            <p class="text-muted">${message}</p>
    `;
    
    if (retryCallback) {
        html += `<button class="btn btn-outline-primary mt-2" onclick="${retryCallback}">重试</button>`;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

function showEmpty(container, message, actionText, actionCallback) {
    let html = `
        <div class="text-center py-5">
            <i class="fas fa-inbox fa-3x text-muted mb-3"></i>
            <p class="text-muted">${message}</p>
    `;
    
    if (actionText && actionCallback) {
        html += `<button class="btn btn-primary mt-2" onclick="${actionCallback}">${actionText}</button>`;
    }
    
    html += '</div>';
    container.innerHTML = html;
}

document.addEventListener('DOMContentLoaded', function() {
    initTheme();
    checkApiStatus();
    initHighlight();
    initKeyboardShortcuts();
    initCodeEditorEnhancements();
    
    const tooltips = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltips.forEach(el => new bootstrap.Tooltip(el));
    
    const popovers = document.querySelectorAll('[data-bs-toggle="popover"]');
    popovers.forEach(el => new bootstrap.Popover(el));
});

window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', e => {
    const config = JSON.parse(localStorage.getItem('tracecoder_config') || '{}');
    if (config.theme === 'auto') {
        document.body.classList.toggle('dark-theme', e.matches);
    }
});

window.TraceCoderUtils = {
    showToast,
    escapeHtml,
    formatTime,
    formatDuration,
    copyToClipboard,
    downloadFile,
    debounce,
    throttle,
    showLoading,
    showError,
    showEmpty,
    animateNumber,
    formatPythonCode,
    initKeyboardShortcuts,
    initCodeEditorEnhancements
};
