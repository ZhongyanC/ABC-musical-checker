from dataclasses import dataclass
from typing import List, Tuple, Dict

@dataclass
class Issue:
    line_index: int
    description: str
    severity: str

class CheckerModule:
    """
    检查与纠正模块基类
    支持检查并可选地执行自动修复逻辑
    """
    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        raise NotImplementedError

class HeaderChecker(CheckerModule):
    """
    头部信息检查与补全模块
    将必须的字段及其默认值作为配置传入
    """
    def __init__(self, required_headers_with_defaults: Dict[str, str]):
        self.required_headers = required_headers_with_defaults

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)
        found_headers = set()
        
        for line in lines:
            if len(line) >= 2 and line[1] == ':':
                found_headers.add(line[0])
                
        for header, default_val in self.required_headers.items():
            if header not in found_headers:
                issues.append(Issue(line_index=-1, description=f"缺失必要的头部字段 {header}", severity="error"))
                if auto_fix:
                    # 自动修复逻辑：在文件开头插入缺失的头部及其默认值
                    modified_lines.insert(0, f"{header}:{default_val}")
                    
        return issues, modified_lines

class VoiceNameFormatter(CheckerModule):
    """
    声部声明格式化模块
    用于检查并修复 V: 标签后多余的空格或不规范缩进
    """
    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = []
        
        for index, line in enumerate(lines):
            clean_line = line.strip()
            
            if clean_line.startswith("V:") and len(line) > len(clean_line):
                issues.append(Issue(line_index=index, description="声部声明存在不规范的缩进或空格", severity="warning"))
                if auto_fix:
                    # 修复逻辑：重新拼接干净的声部字符串
                    voice_name = clean_line[2:].strip()
                    modified_lines.append(f"V:{voice_name}")
                    continue
                    
            modified_lines.append(line)
            
        return issues, modified_lines

class ABCProcessor:
    """
    核心处理引擎
    管理所有注册的模块并按顺序执行检查与修复管道
    """
    def __init__(self):
        self.modules: List[CheckerModule] = []

    def register_module(self, module: CheckerModule):
        self.modules.append(module)

    def run_pipeline(self, abc_content: str, auto_fix: bool = False) -> Tuple[List[Issue], str]:
        lines = abc_content.split("\n")
        all_issues = []
        
        for module in self.modules:
            issues, lines = module.process(lines, auto_fix)
            all_issues.extend(issues)
            
        final_content = "\n".join(lines)
        return all_issues, final_content

if __name__ == "__main__":
    sample_abc = """T:Sample Piece
  V:  1
C D E F | G A B c |
V:2
C, D, E, F, |
"""

    engine = ABCProcessor()
    
    # 规则和默认值通过字典动态配置
    header_rules = {
        "X": "1",
        "T": "Untitled",
        "M": "4/4",
        "L": "1/4"
    }
    
    engine.register_module(HeaderChecker(required_headers_with_defaults=header_rules))
    engine.register_module(VoiceNameFormatter())
    
    print("执行纯检查模式：")
    issues, _ = engine.run_pipeline(sample_abc, auto_fix=False)
    for issue in issues:
        print(f"行号 {issue.line_index} 描述 {issue.description}")
        
    print("\n执行自动修复模式：")
    issues, fixed_abc = engine.run_pipeline(sample_abc, auto_fix=True)
    print("修复后的乐谱内容：")
    print(fixed_abc)