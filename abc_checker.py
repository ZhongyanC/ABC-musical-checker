import re
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Tuple, Dict, Optional

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

class LengthUnifier(CheckerModule):
    """
    统一各声部的 L: 单位音符时值，以所有声部中最小（最精细）的 L: 为目标。

    算法（参考 ABC v2.1 §3.1.7 和 §4.2）：
      1. 收集全局 L:（K: 之前）及各 V: 块内的 L: 行，
         未声明者按 ABC 标准从 M: 推导默认值（M<0.75 → 1/16，否则 1/8）。
      2. 找出最小 L: 作为目标值 target_L。
      3. 对每个有效 L: ≠ target_L 的声部，将乐谱行内全部音符/休止符/和弦
         的时值修饰符乘以 (old_L / target_L)；同步更新其 L: 行。
      4. 更新或插入全局 L: 为 target_L。
      5. inline [L:...] 字段一并替换为 target_L。

    音符时值缩放遵循 ABC 标准时值语法：
      - 裸音符 A  → 乘以 scale（如 scale=2: A → A2）
      - A2 → A4（整数乘）
      - A/ 或 A/2 → A1（即 A）（1/2 * 2 = 1）
      - A3/2 → A3（3/2 * 2 = 3）
      - 和弦 [CEG]2 → [CEG]4（外部修饰符）
      - 和弦 [C2E2G2] → [C4E4G4]（内部修饰符，仅当无外部）
      - Z/X（多小节休止）的计数不缩放
      - 连音符规格 (p:q:r) 不缩放（各音符时值单独处理）
    """

    def _parse_meter(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in lines:
            s = line.lstrip()
            if not s.startswith('M:'):
                continue
            val = s[2:].strip()
            if val in ('C', 'c'):    return (4, 4)
            if val in ('C|', 'c|'): return (2, 2)
            m = re.match(r'(\d+)/(\d+)', val)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _parse_l(self, s: str) -> Optional[Fraction]:
        m = re.match(r'\s*L:\s*(\d+)/(\d+)', s)
        return Fraction(int(m.group(1)), int(m.group(2))) if m else None

    def _default_l(self, meter: Optional[Tuple[int, int]]) -> Fraction:
        if meter is None:
            return Fraction(1, 8)
        return Fraction(1, 16) if (meter[0] / meter[1]) < 0.75 else Fraction(1, 8)

    def _read_len(self, s: str, pos: int) -> Tuple[Fraction, int]:
        """解析 pos 处的时值修饰符，返回 (Fraction, new_pos)。"""
        n, j = len(s), pos
        num_s = ''
        while j < n and s[j].isdigit(): num_s += s[j]; j += 1
        slashes = 0
        while j < n and s[j] == '/': slashes += 1; j += 1
        den_s = ''
        while j < n and s[j].isdigit(): den_s += s[j]; j += 1
        if not num_s and not slashes:
            return Fraction(1), pos
        num = int(num_s) if num_s else 1
        den = (int(den_s) if den_s else 2 ** slashes) if slashes else 1
        return Fraction(num, den), j

    def _len_str(self, f: Fraction) -> str:
        """Fraction → ABC 时值修饰符字符串。"""
        if f == 1:             return ''
        if f.denominator == 1: return str(f.numerator)
        if f.numerator == 1:   return f'/{f.denominator}'
        return f'{f.numerator}/{f.denominator}'

    def _rewrite(self, s: str, scale: Fraction,
                 target_l: Optional[Fraction] = None) -> str:
        """按 scale 缩放字符串 s 中所有音符时值修饰符。"""
        if scale == 1:
            return s
        out: List[str] = []
        i, n = 0, len(s)

        while i < n:
            c = s[i]

            # % 注释 → 保留至行尾
            if c == '%':
                j = s.find('\n', i)
                end = n if j == -1 else j
                out.append(s[i:end]); i = end; continue

            # "..." 和弦符号/注释字符串
            if c == '"':
                j = i + 1
                while j < n and s[j] != '"': j += 1
                if j < n: j += 1
                out.append(s[i:j]); i = j; continue

            # !...! 装饰
            if c == '!':
                j = i + 1
                while j < n and s[j] != '!': j += 1
                if j < n: j += 1
                out.append(s[i:j]); i = j; continue

            # +...+ 装饰
            if c == '+':
                j = i + 1
                while j < n and s[j] != '+': j += 1
                if j < n: j += 1
                out.append(s[i:j]); i = j; continue

            # {...} 装饰音 — 内部也缩放（保持内部比例一致）
            if c == '{':
                j = i + 1
                while j < n and s[j] != '}': j += 1
                inside = s[i+1:j]
                if j < n: j += 1
                out.append('{' + self._rewrite(inside, scale, target_l) + '}')
                i = j; continue

            # inline field [X:...]
            if (c == '[' and i + 2 < n
                    and s[i+1].isalpha() and s[i+2] == ':'):
                j = i + 1
                while j < n and s[j] != ']': j += 1
                inner = s[i+1:j]
                if j < n: j += 1
                if inner.startswith('L:') and target_l is not None:
                    # 替换为目标 L:
                    out.append(
                        f'[L:{target_l.numerator}/{target_l.denominator}]'
                    )
                else:
                    out.append(s[i:j])
                i = j; continue

            # 和弦 [notes]outer_len
            if c == '[':
                # [1 [2 反复标记
                if i + 1 < n and s[i+1].isdigit():
                    out.append(c); i += 1; continue
                j = i + 1
                while j < n and s[j] != ']': j += 1
                inside = s[i+1:j]
                if j < n: j += 1
                outer_len, j = self._read_len(s, j)
                if outer_len != 1:
                    # 有外部修饰符：只缩放外部（避免与内部双重计数）
                    out.append('[' + inside + ']')
                    out.append(self._len_str(outer_len * scale))
                else:
                    # 无外部修饰符：缩放内部各音符
                    out.append('[' + self._rewrite(inside, scale, target_l) + ']')
                i = j; continue

            # Z/X 多小节休止：数字是小节计数，不是 L 单位，不缩放
            if c in 'ZX':
                out.append(c); i += 1
                while i < n and s[i].isdigit():
                    out.append(s[i]); i += 1
                continue

            # (p:q:r 连音符规格 — 仅跳过规格本身，后续各音符单独处理
            if c == '(':
                out.append(c); i += 1
                while i < n and s[i].isdigit(): out.append(s[i]); i += 1
                while i < n and s[i] == ':':
                    out.append(s[i]); i += 1
                    while i < n and s[i].isdigit(): out.append(s[i]); i += 1
                continue

            # 音符/休止符：[变音符]* 音高 [八度]* [时值修饰符]
            if c in '^_=ABCDEFGabcdefgzx':
                start = i
                while i < n and s[i] in '^_=': i += 1
                if i < n and s[i] in 'ABCDEFGabcdefgzx':
                    i += 1
                while i < n and s[i] in ",'": i += 1
                head_end = i
                old_len, i = self._read_len(s, i)
                out.append(s[start:head_end])
                out.append(self._len_str(old_len * scale))
                continue

            out.append(c); i += 1

        return ''.join(out)

    @staticmethod
    def _is_hdr(s: str) -> bool:
        return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        k_idx = next(
            (i for i, l in enumerate(lines) if l.lstrip().startswith('K:')), None
        )
        if k_idx is None:
            return issues, modified_lines

        meter = self._parse_meter(lines[:k_idx + 1])
        default_l = self._default_l(meter)

        # 全局 L:（K: 之前）
        global_l: Optional[Fraction] = None
        global_l_idx: Optional[int] = None
        for i, line in enumerate(lines[:k_idx + 1]):
            val = self._parse_l(line.lstrip())
            if val is not None:
                global_l = val
                global_l_idx = i

        # 扫描各 V: 段
        # 每段记录：v_idx（V: 行索引）、l_idx/l_val（段内 L: 行）、music_idxs（乐谱行）
        segments: List[Dict] = []
        cur: Optional[Dict] = None

        for i in range(k_idx + 1, len(lines)):
            s = lines[i].lstrip()
            if s.startswith('V:'):
                if cur is not None:
                    segments.append(cur)
                cur = {
                    'v_idx': i, 'l_idx': None, 'l_val': None,
                    'music_idxs': [], 'in_header': True,
                }
            elif cur is not None:
                if cur['in_header'] and s.startswith('L:'):
                    cur['l_idx'] = i
                    cur['l_val'] = self._parse_l(s)
                elif not s or s.startswith('%'):
                    pass  # 空行 / 注释 / %% 指令
                elif self._is_hdr(s):
                    pass  # 其他头部字段（K:、M: 等）
                else:
                    cur['in_header'] = False
                    cur['music_idxs'].append(i)

        if cur is not None:
            segments.append(cur)

        if not segments:
            return issues, modified_lines

        # 计算各段有效 L:
        for seg in segments:
            seg['eff_l'] = seg['l_val'] if seg['l_val'] is not None else (
                global_l if global_l is not None else default_l
            )

        # 只考虑有实际乐谱行的段
        active = [seg for seg in segments if seg['music_idxs']]
        if not active:
            return issues, modified_lines

        all_l = {seg['eff_l'] for seg in active}
        if global_l is not None:
            all_l.add(global_l)

        min_l = min(all_l)

        if all(seg['eff_l'] == min_l for seg in active):
            return issues, modified_lines

        # 报告问题
        voice_info = ', '.join(
            f"V:{lines[seg['v_idx']].lstrip()[2:].strip().split()[0]}"
            f"→L:{seg['eff_l'].numerator}/{seg['eff_l'].denominator}"
            for seg in active
        )
        issues.append(Issue(
            line_index=k_idx + 1,
            description=(
                f"各声部 L: 不统一（{voice_info}）；"
                f"将统一为最小值 L:{min_l.numerator}/{min_l.denominator}"
            ),
            severity="warning",
        ))

        if not auto_fix:
            return issues, modified_lines

        # 修复：缩放各段乐谱 + 更新 L: 行
        for seg in active:
            scale = seg['eff_l'] / min_l
            if scale != 1:
                for mi in seg['music_idxs']:
                    modified_lines[mi] = self._rewrite(
                        modified_lines[mi], scale, min_l
                    )
            if seg['l_idx'] is not None:
                modified_lines[seg['l_idx']] = re.sub(
                    r'L:\s*\d+/\d+',
                    f'L:{min_l.numerator}/{min_l.denominator}',
                    modified_lines[seg['l_idx']],
                )

        # 更新或插入全局 L:
        new_l_str = f'L:{min_l.numerator}/{min_l.denominator}'
        if global_l_idx is not None:
            modified_lines[global_l_idx] = re.sub(
                r'L:\s*\d+/\d+', new_l_str, modified_lines[global_l_idx]
            )
        else:
            modified_lines.insert(k_idx, new_l_str)

        return issues, modified_lines


class ClefAutoSelector(CheckerModule):
    """
    自动判断音区并选择合适的谱号 (treble / bass)。

    依据 ABC standard v2.1 §4.1 (Pitch) 与 §4.6 (Clefs and transposition):
      - 大写 C..B  对应中央 C 起的一个八度  (C = MIDI 60)
      - 小写 c..b  对应再上一个八度        (c = MIDI 72)
      - 每个 ','  降低一个八度,  每个 "'" 升高一个八度
      - clef 可写在 K: 或 V: 行, 形如 'clef=bass' 或裸名 'bass'

    策略:
      1. 解析每个声部 (V:) 的全部音符为 MIDI 数值
      2. 统计落在 treble / bass 各自舒适音域之外 (即需要超过若干加线)
         的音符数量
      3. 若另一个谱号能显著减少加线越界数量, 则建议 / 自动替换
         声部的默认谱号 (V: / K: 行)
      4. 进一步按小节扫描音区, 若出现持续若干小节的音区临时切换,
         则在相应小节前插入 inline clef 字段 '[K:clef=xxx]'
         (§3.1 inline field + §4.6 clef)
    """

    # ABC 音名 -> MIDI (基准八度)
    BASE_PITCHES = {
        'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
        'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83,
    }

    # 谱号舒适音区 (允许大致 ledger_limit 条加线之内)
    # treble: 五线谱内 E4(64) - F5(77); 加 2 加线后大约 A3(57) - C6(84)
    # bass:   五线谱内 G2(43) - A3(57); 加 2 加线后大约 E2(40) - E4(64)
    CLEF_RANGES = {
        'treble': (55, 84),
        'bass':   (40, 64),
    }

    KNOWN_CLEFS = ('treble', 'bass', 'alto', 'tenor', 'perc', 'none')

    NOTE_RE = re.compile(r"(\^\^|__|\^|_|=)?([A-Ga-g])([,']*)")

    def __init__(self, ledger_threshold_ratio: float = 0.25,
                 min_improvement: int = 1,
                 min_run_length: int = 2):
        """
        ledger_threshold_ratio:
            当前谱号下越界音符占比超过该值才考虑换谱号
        min_improvement:
            候选谱号必须比当前谱号至少减少这么多越界音符才会被采用
        min_run_length:
            小节级临时切换时, 偏好新谱号的连续小节数必须达到该值,
            否则认为是偶发而不插入 inline clef 标记
        """
        self.ledger_threshold_ratio = ledger_threshold_ratio
        self.min_improvement = min_improvement
        self.min_run_length = min_run_length

    # ---------- 解析辅助 ----------

    def _strip_non_music(self, line: str) -> str:
        """剔除非音符 token, 防止误把和弦符号 / 注释里的字母当成音符"""
        line = re.sub(r'%.*$', '', line)               # 行注释
        line = re.sub(r'"[^"]*"', '', line)            # 和弦符号 / 注释字符串
        line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', line)  # inline field
        line = re.sub(r'!\w+!', '', line)              # !decoration!
        line = re.sub(r'\+[\w]+\+', '', line)          # +decoration+
        line = re.sub(r'\{[^}]*\}', '', line)          # 装饰音
        return line

    def _note_to_midi(self, match) -> int:
        letter = match.group(2)
        octave_marks = match.group(3) or ''
        midi = self.BASE_PITCHES[letter]
        midi += 12 * octave_marks.count("'")
        midi -= 12 * octave_marks.count(',')
        return midi

    def _extract_pitches(self, music_lines: List[str]) -> List[int]:
        pitches = []
        for line in music_lines:
            cleaned = self._strip_non_music(line)
            for m in self.NOTE_RE.finditer(cleaned):
                pitches.append(self._note_to_midi(m))
        return pitches

    def _out_of_range(self, pitches: List[int], clef: str) -> int:
        low, high = self.CLEF_RANGES[clef]
        return sum(1 for p in pitches if p < low or p > high)

    def _suggest_clef(self, pitches: List[int], current: str) -> Optional[str]:
        if not pitches:
            return None
        if current not in self.CLEF_RANGES:
            current = 'treble'
        cur_out = self._out_of_range(pitches, current)
        # 越界比例不大就保持现状
        if cur_out / len(pitches) < self.ledger_threshold_ratio:
            return None
        # 在 treble / bass 中挑越界最少的
        best = min(self.CLEF_RANGES.keys(), key=lambda c: self._out_of_range(pitches, c))
        if best == current:
            return None
        if cur_out - self._out_of_range(pitches, best) < self.min_improvement:
            return None
        return best

    # ---------- clef 字段读写 ----------

    def _extract_clef(self, line: str) -> Optional[str]:
        m = re.search(r'clef\s*=\s*([A-Za-z0-9+\-]+)', line)
        if m:
            name = m.group(1)
            for c in self.KNOWN_CLEFS:
                if name.startswith(c):
                    return c
            return None
        # 裸的谱号名 (例如 'K:C bass' 或 'V:1 treble')
        head = re.match(r'\s*[KV]:\s*(.*)', line)
        if head:
            for tok in head.group(1).split():
                if tok in self.KNOWN_CLEFS:
                    return tok
        return None

    def _set_clef(self, line: str, clef: str) -> str:
        if re.search(r'clef\s*=\s*[A-Za-z0-9+\-]+', line):
            return re.sub(r'clef\s*=\s*[A-Za-z0-9+\-]+', f'clef={clef}', line)
        for c in ('treble', 'bass', 'alto', 'tenor'):
            if re.search(rf'(?<![A-Za-z]){c}(?![A-Za-z])', line):
                return re.sub(rf'(?<![A-Za-z]){c}(?![A-Za-z])', clef, line, count=1)
        return line.rstrip() + f' clef={clef}'

    # ---------- 主流程 ----------

    @staticmethod
    def _is_header_line(line: str) -> bool:
        s = line.lstrip()
        return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues: List[Issue] = []
        modified_lines = list(lines)

        # 找到第一个 K: 行 — ABC 规定 K: 之后才是乐谱主体
        body_start = None
        k_line_idx = None
        for i, l in enumerate(lines):
            if l.lstrip().startswith('K:'):
                k_line_idx = i
                body_start = i + 1
                break
        if body_start is None:
            return issues, modified_lines

        global_clef = self._extract_clef(lines[k_line_idx]) or 'treble'

        # 把 body 按 V: 切分成段; 没出现 V: 则整体当作一段
        segments: List[Tuple[Optional[int], List[int]]] = []
        current_voice_idx: Optional[int] = None
        current_music: List[int] = []

        def flush():
            if current_voice_idx is not None or current_music:
                segments.append((current_voice_idx, list(current_music)))

        for i in range(body_start, len(lines)):
            line = lines[i]
            if line.lstrip().startswith('V:'):
                flush()
                current_voice_idx = i
                current_music = []
            elif line.strip() and not self._is_header_line(line):
                current_music.append(i)
        flush()

        if not segments:
            return issues, modified_lines

        for voice_idx, music_idxs in segments:
            music_text = [lines[i] for i in music_idxs]
            pitches = self._extract_pitches(music_text)
            if not pitches:
                continue

            if voice_idx is not None:
                current_clef = self._extract_clef(lines[voice_idx]) or global_clef
                target_idx = voice_idx
            else:
                current_clef = global_clef
                target_idx = k_line_idx

            # --- 声部级: 整体重选谱号 ---
            suggested = self._suggest_clef(pitches, current_clef)
            effective_clef = current_clef
            if suggested is not None:
                cur_out = self._out_of_range(pitches, current_clef)
                new_out = self._out_of_range(pitches, suggested)
                desc = (f"音区更适合 {suggested} 谱号 "
                        f"(当前 {current_clef}: 越界 {cur_out}/{len(pitches)}, "
                        f"建议后越界 {new_out}/{len(pitches)})")
                issues.append(Issue(line_index=target_idx, description=desc, severity="info"))
                if auto_fix:
                    modified_lines[target_idx] = self._set_clef(modified_lines[target_idx], suggested)
                effective_clef = suggested

            # --- 小节级: 检测临时音区变化, 插入 inline clef ---
            self._process_inline_clefs(
                lines=lines,
                modified_lines=modified_lines,
                music_idxs=music_idxs,
                starting_clef=effective_clef,
                auto_fix=auto_fix,
                issues=issues,
            )

        return issues, modified_lines

    # ---------- 小节级 inline clef 处理 ----------

    def _best_clef_for(self, pitches: List[int]) -> Optional[str]:
        if not pitches:
            return None
        return min(self.CLEF_RANGES.keys(),
                   key=lambda c: self._out_of_range(pitches, c))

    def _process_inline_clefs(self,
                              lines: List[str],
                              modified_lines: List[str],
                              music_idxs: List[int],
                              starting_clef: str,
                              auto_fix: bool,
                              issues: List[Issue]) -> None:
        """扫描声部内各小节的音区, 必要时插入 [K:clef=xxx] inline 标记"""

        # 按单个 '|' 切分每一行, 偶数索引是小节内容, 奇数索引是小节线
        line_chunks: Dict[int, List[str]] = {}
        measures: List[Dict] = []

        for li in music_idxs:
            parts = re.split(r'(\|)', lines[li])
            line_chunks[li] = parts
            for ci in range(0, len(parts), 2):
                content = parts[ci]
                cleaned = self._strip_non_music(content)
                pitches = [self._note_to_midi(m)
                           for m in self.NOTE_RE.finditer(cleaned)]
                # 纯空白 / 纯小节线装饰 (例如开头空 chunk) 跳过
                if not cleaned.strip() and not pitches:
                    continue
                measures.append({
                    'line_idx': li,
                    'chunk_idx': ci,
                    'pitches': pitches,
                })

        if len(measures) < self.min_run_length + 1:
            return

        m_clefs = [self._best_clef_for(m['pitches']) for m in measures]

        if starting_clef not in self.CLEF_RANGES:
            starting_clef = 'treble'
        active = starting_clef

        inserts: List[Tuple[int, int, str]] = []  # (line_idx, chunk_idx, clef)

        i = 0
        while i < len(measures):
            c = m_clefs[i]
            # 无音符 / 已经是当前谱号 -> 跳过
            if c is None or c == active:
                i += 1
                continue
            # 向前看: 统计连续偏好 c 的小节 (允许中间夹空小节)
            j = i
            run = 0
            while j < len(measures) and m_clefs[j] in (c, None):
                if m_clefs[j] == c:
                    run += 1
                j += 1
            if run < self.min_run_length:
                i = j
                continue
            # 验证切换确有改善
            run_pitches: List[int] = []
            for k in range(i, j):
                run_pitches.extend(measures[k]['pitches'])
            if (self._out_of_range(run_pitches, active)
                    - self._out_of_range(run_pitches, c)) < self.min_improvement:
                i = j
                continue
            inserts.append((measures[i]['line_idx'],
                            measures[i]['chunk_idx'], c))
            active = c
            i = j

        if not inserts:
            return

        for li, _, c in inserts:
            issues.append(Issue(
                line_index=li,
                description=f"音区临时切换, 建议在该小节前插入 inline 谱号 [K:clef={c}]",
                severity="info",
            ))

        if not auto_fix:
            return

        # 应用插入 (从后往前以保持 chunk 索引稳定)
        inserts_sorted = sorted(inserts,
                                key=lambda x: (x[0], x[1]),
                                reverse=True)
        for li, ci, c in inserts_sorted:
            chunks = line_chunks[li]
            original = chunks[ci]
            stripped = original.lstrip()
            leading = original[:len(original) - len(stripped)]
            marker = f'[K:clef={c}]'
            # 保留原有前导空白, 紧接 marker, 再接原内容
            chunks[ci] = f'{leading}{marker}{stripped}' if stripped else f'{leading}{marker}'

        for li, chunks in line_chunks.items():
            modified_lines[li] = ''.join(chunks)


class VoiceBarCountChecker(CheckerModule):
    """
    检查各声部 (V:) 的小节数是否一致。
    以最多小节数的声部为基准，对声部数不足的声部报告 warning。
    auto_fix=True 时，在不足的声部末尾补齐空白小节，
    填充内容为 x 音符，数量由 M: 与 L: 计算得出。
    """

    # 匹配所有合法的小节线变体，统一归一为 '|'
    _BAR_RE = re.compile(r':\|:|:\||\|\||\|\]|\[?\|:?')

    def _count_measures(self, music_lines: List[str]) -> int:
        content = ' '.join(music_lines)
        content = re.sub(r'%.*$', '', content, flags=re.MULTILINE)   # 行注释
        content = re.sub(r'"[^"]*"', '', content)                    # 和弦/注释字符串
        content = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content)        # inline field
        content = self._BAR_RE.sub('|', content)                     # 归一化小节线
        parts = content.split('|')
        return sum(1 for p in parts if p.strip())

    def _notes_per_measure(self, lines: List[str]) -> int:
        """
        根据 M: 与 L: 计算每小节内含多少个 L 时值单位。
        公式: (M_分子/M_分母) / (L_分子/L_分母) = M_分子 * L_分母 / (M_分母 * L_分子)
        例: M:4/4, L:1/4 -> (4*4)/(4*1) = 4
        """
        meter = unit = None
        for line in lines:
            if re.match(r'\s*M:\s*\d', line):
                m = re.match(r'\s*M:\s*(\d+)/(\d+)', line)
                if m:
                    meter = (int(m.group(1)), int(m.group(2)))
            if re.match(r'\s*L:\s*\d', line):
                m = re.match(r'\s*L:\s*(\d+)/(\d+)', line)
                if m:
                    unit = (int(m.group(1)), int(m.group(2)))
        if meter is None or unit is None:
            return 4  # 默认回退
        numerator = meter[0] * unit[1]
        denominator = meter[1] * unit[0]
        return max(1, round(numerator / denominator))

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        # 找到 K: 行，乐谱主体从其后一行开始
        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1
                break
        if body_start is None:
            return issues, modified_lines

        notes_per_measure = self._notes_per_measure(lines[:body_start])

        # 按 V: 切分声部
        # voices: vid -> (v_line_idx, last_music_line_idx, music_lines)
        voices: Dict[str, Tuple[int, int, List[str]]] = {}
        current_id = '__default__'
        current_v_idx = body_start
        current_last_music_idx = body_start
        current_music: List[str] = []

        def _flush():
            if current_music:
                voices[current_id] = (current_v_idx, current_last_music_idx, list(current_music))

        for i in range(body_start, len(lines)):
            line = lines[i]
            stripped = line.lstrip()
            if stripped.startswith('V:'):
                _flush()
                current_music.clear()
                voice_body = stripped[2:].strip()
                current_id = voice_body.split()[0] if voice_body else str(i)
                current_v_idx = i
                current_last_music_idx = i
            elif stripped and not (len(stripped) >= 2 and stripped[1] == ':' and stripped[0].isalpha()):
                current_music.append(line)
                current_last_music_idx = i

        _flush()
        voices.pop('__default__', None)

        if len(voices) < 2:
            return issues, modified_lines

        counts = {vid: (v_idx, last_idx, self._count_measures(music))
                  for vid, (v_idx, last_idx, music) in voices.items()}

        all_counts = [c for _, _, c in counts.values()]
        if len(set(all_counts)) == 1:
            return issues, modified_lines

        max_count = max(all_counts)

        for vid, (v_idx, last_idx, count) in counts.items():
            if count >= max_count:
                continue

            missing = max_count - count
            issues.append(Issue(
                line_index=v_idx,
                description=(f"声部 V:{vid} 共 {count} 个小节，"
                             f"少于最长声部的 {max_count} 个（缺 {missing} 小节）"),
                severity="warning",
            ))

            if auto_fix:
                measure_content = f'x{notes_per_measure}'
                fill_parts = ' | '.join([measure_content] * missing)
                last_line = modified_lines[last_idx].rstrip()
                if last_line.endswith('|'):
                    modified_lines[last_idx] = last_line + ' ' + fill_parts + ' |'
                else:
                    modified_lines[last_idx] = last_line + ' | ' + fill_parts + ' |'

        return issues, modified_lines


class MeasureDurationChecker(CheckerModule):
    """
    检查各声部每个小节的时值总和是否符合 M: 与该声部默认时值 L: 的设定。

    流程：
      1. 调用 VoiceBarCountChecker（纯检查模式）确认各声部小节数一致；
         若不一致则直接返回小节数问题，不继续检查。
      2. 解析头部 M: 与全局 L:（仅作回退）；每个带 V: 的声部在 V: 至首行乐谱之间扫描 L:，
         未写则使用全局 L:。按各声部自身的 L: 计算每小节期望的 L 单位数。
      3. 逐声部、逐小节解析所有音符时值，与期望值比对。

    auto_fix=True 时：
      - 不足时值的小节末尾补 z 休止符；
      - 超出时值的小节从末尾逐音删除直至 ≤ 期望，再按不足条件补休止；
      - 弱起检测：若所有声部第一小节在全音符尺度下均短于拍号且彼此相等，识别为弱起，
        跳过全部第一小节；否则第一小节对齐到「全音符下最长」再换算回各声部 L 单位修复。
    """

    _BAR_RE            = re.compile(r':\|:|:\||\|\||\|\]|\[?\|:?')
    _TUPLET_Q          = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}
    _TUPLET_Q_COMPOUND = {2: 3, 3: 2, 4: 3, 5: 3, 6: 2, 7: 3, 8: 3, 9: 3}

    # ---------- 头部解析 ----------

    def _parse_headers(self, lines: List[str]) -> Tuple[Optional[Tuple[int, int]], Optional[Tuple[int, int]]]:
        meter = unit = None
        for line in lines:
            s = line.lstrip()
            if s.startswith('M:'):
                val = s[2:].strip()
                if val in ('C', 'c'):        meter = (4, 4)
                elif val in ('C|', 'c|'):    meter = (2, 2)
                elif val.lower() == 'none':  meter = None
                else:
                    m = re.match(r'\(?([\d+]+)\)?/(\d+)', val)
                    if m:
                        num = sum(int(x) for x in m.group(1).split('+'))
                        meter = (num, int(m.group(2)))
            if s.startswith('L:'):
                m = re.match(r'L:\s*(\d+)/(\d+)', s)
                if m:
                    unit = (int(m.group(1)), int(m.group(2)))
        if unit is None and meter is not None:
            ratio = meter[0] / meter[1]
            unit = (1, 16) if ratio < 0.75 else (1, 8)
        return meter, unit

    @staticmethod
    def _parse_l_line(line: str) -> Optional[Tuple[int, int]]:
        s = line.lstrip()
        if not s.startswith('L:'):
            return None
        m = re.match(r'L:\s*(\d+)/(\d+)', s)
        if not m:
            return None
        return int(m.group(1)), int(m.group(2))

    def _scan_voice_l_unit(
        self,
        lines: List[str],
        v_line_idx: int,
        first_music_line_idx: Optional[int],
        next_v_line_idx: Optional[int],
    ) -> Optional[Tuple[int, int]]:
        """
        在 V: 行之后、该声部首行乐谱之前扫描 L:（中间可含 %%、空行、注释等）。
        若未找到则返回 None，由调用方回退到全局 L:。
        """
        start = v_line_idx + 1
        end = len(lines)
        if next_v_line_idx is not None:
            end = min(end, next_v_line_idx)
        if first_music_line_idx is not None:
            end = min(end, first_music_line_idx)
        for j in range(start, end):
            raw = lines[j]
            s = raw.lstrip()
            if not s:
                continue
            if s.startswith('%'):
                continue
            if s.startswith('%%'):
                continue
            if s.startswith('V:'):
                break
            parsed = self._parse_l_line(s)
            if parsed is not None:
                return parsed
        return None

    def _is_compound(self, mn: int, md: int) -> bool:
        return md == 8 and mn in (6, 9, 12)

    # ---------- 清理 ----------

    def _clean(self, s: str) -> str:
        s = re.sub(r'%.*$', '', s, flags=re.MULTILINE)
        s = re.sub(r'"[^"]*"', '', s)
        s = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', s)
        s = re.sub(r'\{/?[^}]*\}', '', s)
        s = re.sub(r'![^!]*!', '', s)
        s = re.sub(r'\+[^+]+\+', '', s)
        return s

    # ---------- 长度修饰符解析 ----------

    def _parse_len(self, s: str, pos: int) -> Tuple[Fraction, int]:
        n, start = len(s), pos
        num_str = ''
        while pos < n and s[pos].isdigit():
            num_str += s[pos]; pos += 1
        slash_cnt = 0
        while pos < n and s[pos] == '/':
            slash_cnt += 1; pos += 1
        den_str = ''
        while pos < n and s[pos].isdigit():
            den_str += s[pos]; pos += 1
        if not num_str and not slash_cnt:
            return Fraction(1), start
        num = int(num_str) if num_str else 1
        den = (int(den_str) if den_str else 2 ** slash_cnt) if slash_cnt else 1
        return Fraction(num, den), pos

    # ---------- 单小节时值计算（共用核心） ----------

    def _calc(self, s: str, is_compound: bool, track_pos: bool = False
              ) -> 'Tuple[Optional[Fraction], List[Tuple[int,int,Fraction]]]':
        """
        计算字符串 s 中所有音符时值之和，并可选地返回每个 token 的位置列表。
        track_pos=True 时在原始字符串中追踪位置（用于截断修复）；
        track_pos=False 时先调用 _clean，仅关心总和（用于检查）。
        返回 (total_or_None, tokens)。
        total=None 表示含多小节休止符 Z/X，跳过。
        """
        if not track_pos:
            s = self._clean(s)
        n = len(s)
        total = Fraction(0)
        pos = 0
        tuplet_stack: List[List] = []
        tokens: List[Tuple[int, int, Fraction]] = []
        # 附点节奏状态（< > broken rhythm）
        next_note_mult = Fraction(1)   # 下一个音符的缩放因子
        last_actual_dur = Fraction(0)  # 上一个音符的实际时值（用于追溯修正）

        def _add(start: int, end: int, dur: Fraction):
            nonlocal total, next_note_mult, last_actual_dur
            actual_dur = dur * next_note_mult
            next_note_mult = Fraction(1)
            if tuplet_stack:
                actual_dur *= tuplet_stack[-1][1]
                tuplet_stack[-1][0] -= 1
                if tuplet_stack[-1][0] <= 0:
                    tuplet_stack.pop()
            total += actual_dur
            last_actual_dur = actual_dur
            if track_pos:
                tokens.append((start, end, actual_dur))

        while pos < n:
            c = s[pos]
            if c in ' \t\n~HLMOPSTuv.': pos += 1; continue
            if c in ')-_&\\':           pos += 1; continue

            # 附点节奏 > 和 < (§4.14)
            # n 个 > ：前音符 ×(2^(n+1)−1)/2^n，后音符 ×1/2^n
            # n 个 < ：前音符 ×1/2^n，后音符 ×(2^(n+1)−1)/2^n
            if c in '<>':
                kind = c
                cnt = 1
                pos += 1
                while pos < n and s[pos] == kind:
                    cnt += 1; pos += 1
                n_pow = 2 ** cnt
                if kind == '>':
                    prev_mult = Fraction(2 * n_pow - 1, n_pow)
                    next_note_mult = Fraction(1, n_pow)
                else:
                    prev_mult = Fraction(1, n_pow)
                    next_note_mult = Fraction(2 * n_pow - 1, n_pow)
                # 追溯修正上一个音符
                if last_actual_dur:
                    delta = last_actual_dur * (prev_mult - 1)
                    total += delta
                    last_actual_dur += delta
                    if track_pos and tokens:
                        s_t, e_t, old = tokens[-1]
                        tokens[-1] = (s_t, e_t, old + delta)
                continue

            # 跳过注释（仅 track_pos 模式，clean 模式已被 _clean 去掉）
            if c == '%':
                while pos < n and s[pos] != '\n': pos += 1
                continue
            # 跳过和弦符号、装饰音、inline 字段（track_pos 模式）
            if c == '"':
                pos += 1
                while pos < n and s[pos] != '"': pos += 1
                if pos < n: pos += 1
                continue
            if c == '{':
                pos += 1
                while pos < n and s[pos] != '}': pos += 1
                if pos < n: pos += 1
                continue
            if c == '!':
                pos += 1
                while pos < n and s[pos] != '!': pos += 1
                if pos < n: pos += 1
                continue
            if c == '+':
                pos += 1
                while pos < n and s[pos] != '+': pos += 1
                if pos < n: pos += 1
                continue
            if (c == '[' and pos + 1 < n and s[pos+1].isalpha()
                    and pos + 2 < n and s[pos+2] == ':'):
                pos += 1
                while pos < n and s[pos] != ']': pos += 1
                if pos < n: pos += 1
                continue

            # 连音符 (§4.13)：(p 或 (p:q 或 (p:q:r 或 (p::r
            # (p 后没有数字 → 是连弧线（slur），无时值效果，直接跳过
            if c == '(':
                pos += 1
                p_str = ''
                while pos < n and s[pos].isdigit(): p_str += s[pos]; pos += 1
                if not p_str: continue          # slur，无效果
                p = int(p_str); q = r = None
                if pos < n and s[pos] == ':':
                    pos += 1
                    q_str = ''
                    while pos < n and s[pos].isdigit(): q_str += s[pos]; pos += 1
                    q = int(q_str) if q_str else None
                    if pos < n and s[pos] == ':':
                        pos += 1
                        r_str = ''
                        while pos < n and s[pos].isdigit(): r_str += s[pos]; pos += 1
                        r = int(r_str) if r_str else None
                tbl = self._TUPLET_Q_COMPOUND if is_compound else self._TUPLET_Q
                if q is None: q = tbl.get(p, 2)
                if r is None: r = p
                if p > 0:
                    tuplet_stack.append([r, Fraction(q, p)])
                continue

            # 和弦
            if c == '[':
                tk_start = pos; pos += 1
                if pos < n and s[pos].isdigit(): pos += 1; continue  # [1 [2
                while pos < n and s[pos] in '^_=': pos += 1
                internal = Fraction(1)
                if pos < n and s[pos] in 'ABCDEFGabcdefgzx':
                    pos += 1
                    # 跳过八度标记（, 和 '），然后解析时值
                    while pos < n and s[pos] in ",'":
                        pos += 1
                    internal, pos = self._parse_len(s, pos)
                while pos < n and s[pos] != ']': pos += 1
                if pos < n: pos += 1
                ext, np2 = self._parse_len(s, pos)
                chord_len = ext if np2 > pos else internal
                pos = np2 if np2 > pos else pos
                _add(tk_start, pos, chord_len)
                continue

            # 临时变音 + 音符/休止符
            if c in '^_=ABCDEFGabcdefgzxZX':
                tk_start = pos
                while pos < n and s[pos] in '^_=': pos += 1
                if pos >= n or s[pos] not in 'ABCDEFGabcdefgzxZX':
                    continue
                nc = s[pos]; pos += 1
                if nc in 'ZX':
                    while pos < n and s[pos].isdigit(): pos += 1
                    return None, []     # 多小节休止符
                # 跳过八度标记 , 和 '（不影响时值）
                while pos < n and s[pos] in ",'":
                    pos += 1
                note_len, pos = self._parse_len(s, pos)
                _add(tk_start, pos, note_len)
                continue

            pos += 1

        return total, tokens

    def _measure_duration(self, measure_str: str, is_compound: bool) -> Optional[Fraction]:
        total, _ = self._calc(measure_str, is_compound, track_pos=False)
        return total

    # ---------- 时值修复辅助 ----------

    @staticmethod
    def _rest_str(f: Fraction) -> str:
        """Fraction → ABC z 休止符写法"""
        num, den = f.numerator, f.denominator
        if den == 1:   return 'z' if num == 1 else f'z{num}'
        if num == 1 and den == 2: return 'z/'
        if num == 1:   return f'z/{den}'
        return f'z{num}/{den}'

    def _fix_measure(self, measure: str, expected: Fraction, is_compound: bool) -> str:
        """
        修复单小节时值：
          不足 → 末尾补 z 休止；
          超出 → 从末尾逐 token 删除直至 ≤ expected，再按不足条件补休止。
        """
        actual = self._measure_duration(measure, is_compound)
        if actual is None or actual == expected:
            return measure

        if actual < expected:
            return measure.rstrip() + ' ' + self._rest_str(expected - actual)

        # 超出：追踪 token 位置，找截断点
        _, tokens = self._calc(measure, is_compound, track_pos=True)
        if not tokens:
            return measure

        cumsum = Fraction(0)
        cut_pos = 0
        for tk_start, tk_end, dur in tokens:
            if cumsum + dur <= expected:
                cumsum += dur
                cut_pos = tk_end
            else:
                break

        fixed = measure[:cut_pos]
        if cumsum < expected:
            fixed = fixed.rstrip() + ' ' + self._rest_str(expected - cumsum)
        return fixed

    def _fix_content(self, content: str, expected: Fraction,
                     is_compound: bool, first_expected: Optional[Fraction]) -> str:
        """
        修复声部内容字符串中所有小节的时值。
        first_expected=None  → 跳过第一小节（弱起，不补足）；
        first_expected=Fraction → 第一小节按该值修复，其余按 expected 修复。
        """
        bar_spans = list(self._BAR_RE.finditer(content))
        segments: List[Tuple[str, str]] = []
        start = 0
        for m in bar_spans:
            segments.append((content[start:m.start()], m.group()))
            start = m.end()
        trailing = content[start:]
        if trailing.strip():
            segments.append((trailing, ''))

        result = []
        for i, (measure, bar) in enumerate(segments):
            if not measure.strip():
                result.append(measure + bar)
                continue
            if i == 0:
                if first_expected is None:
                    result.append(measure + bar)   # 弱起：原样保留
                    continue
                exp = first_expected
            else:
                exp = expected
            fixed = self._fix_measure(measure, exp, is_compound)
            sep = ' ' if (fixed and bar and not fixed.endswith(' ')) else ''
            result.append(fixed + sep + bar)
        return ''.join(result)

    # ---------- 主流程 ----------

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []

        # Step 1: 前置小节数一致性检查
        bar_issues, _ = VoiceBarCountChecker().process(lines, auto_fix=False)
        if bar_issues:
            for bi in bar_issues:
                bi.description = '[前置检查] 小节数不一致，已跳过时值检查：' + bi.description
            return bar_issues, list(lines)

        # Step 2: 定位 K:
        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1; break
        if body_start is None:
            return issues, list(lines)

        # Step 3: 解析 M: 与全局 L:（全局 L 仅作声部未写 L: 时的回退）
        meter, global_unit = self._parse_headers(lines[:body_start])
        if meter is None or global_unit is None:
            return issues, list(lines)
        bar_semibreves = Fraction(meter[0], meter[1])
        is_cmpd = self._is_compound(meter[0], meter[1])

        # Step 4: 切分声部，记录行索引
        voices: Dict[str, Tuple[int, List[int]]] = {}
        cur_id, cur_v_idx = '__default__', body_start
        cur_idxs: List[int] = []

        def _flush():
            if cur_idxs:
                voices[cur_id] = (cur_v_idx, list(cur_idxs))

        def _is_hdr(s: str) -> bool:
            return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

        for i in range(body_start, len(lines)):
            s = lines[i].lstrip()
            if s.startswith('V:'):
                _flush(); cur_idxs = []
                vb = s[2:].strip()
                cur_id = vb.split()[0] if vb else str(i)
                cur_v_idx = i
            elif s and not _is_hdr(s) and not s.startswith('%'):
                cur_idxs.append(i)
        _flush()
        voices.pop('__default__', None)

        if not voices:
            body_idxs = [i for i in range(body_start, len(lines))
                         if lines[i].lstrip() and not _is_hdr(lines[i].lstrip())]
            if body_idxs:
                voices['1'] = (body_start, body_idxs)

        # 各声部默认时值 L:（V: 块内扫描，否则用全局）
        v_ordered = sorted(voices.items(), key=lambda kv: kv[1][0])
        voice_units: Dict[str, Tuple[int, int]] = {}
        expected_by_vid: Dict[str, Fraction] = {}
        for k, (vid, (v_idx, idxs)) in enumerate(v_ordered):
            next_v_line = v_ordered[k + 1][1][0] if k + 1 < len(v_ordered) else None
            first_music = idxs[0] if idxs else None
            scanned = self._scan_voice_l_unit(lines, v_idx, first_music, next_v_line)
            u = scanned if scanned is not None else global_unit
            voice_units[vid] = u
            expected_by_vid[vid] = bar_semibreves / Fraction(u[0], u[1])

        modified_lines = list(lines)

        # ---------- 弱起检测（第一小节时值换算到全音符比例后跨声部比较）----------
        def _first_measure_dur(idxs: List[int]) -> Optional[Fraction]:
            if not idxs: return None
            content = ' '.join(lines[i] for i in idxs)
            content = re.sub(r'"[^"]*"', '', re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content))
            segs = [s for s in self._BAR_RE.split(content) if s.strip()]
            return self._measure_duration(segs[0], is_cmpd) if segs else None

        first_durs: Dict[str, Optional[Fraction]] = {
            vid: _first_measure_dur(idxs)
            for vid, (_, idxs) in voices.items()
        }

        first_semibreves: Dict[str, Fraction] = {}
        for vid, d in first_durs.items():
            if d is None:
                continue
            u = voice_units[vid]
            first_semibreves[vid] = d * Fraction(u[0], u[1])

        voices_with_first = [vid for vid in voices if first_durs.get(vid) is not None]
        is_global_pickup = (
            len(voices) > 0
            and len(voices_with_first) == len(voices)
            and len(first_semibreves) == len(voices)
            and len(set(first_semibreves.values())) == 1
            and next(iter(first_semibreves.values())) != bar_semibreves
        )

        # 非弱起时：第一小节以「全音符下最长者」为基准，再换算回各声部自身 L 单位
        max_first_by_vid: Dict[str, Optional[Fraction]] = {}
        if not is_global_pickup:
            if first_semibreves:
                max_s = max(first_semibreves.values())
            else:
                max_s = bar_semibreves
            for vid in voices:
                u = voice_units[vid]
                max_first_by_vid[vid] = max_s / Fraction(u[0], u[1])
        else:
            for vid in voices:
                max_first_by_vid[vid] = None

        # ---------- 检查 + 修复 ----------
        for vid, (v_idx, idxs) in voices.items():
            if not idxs:
                continue

            content_orig = ' '.join(lines[i] for i in idxs)
            # 预处理（仅用于小节分割和检查，不影响 _fix_content）
            pre = re.sub(r'"[^"]*"', '',
                  re.sub(r'\[[A-Za-z]:[^\]]*\]', '', content_orig))
            measures = [m for m in self._BAR_RE.sub('|', pre).split('|')]
            exp_voice = expected_by_vid[vid]

            for m_num, measure in enumerate(measures, start=1):
                if not measure.strip():
                    continue
                actual = self._measure_duration(measure, is_cmpd)
                if actual is None:
                    continue

                if m_num == 1:
                    if is_global_pickup:
                        continue   # 弱起，跳过检查
                    ref = max_first_by_vid.get(vid)
                    if ref is None or actual == ref:
                        continue
                    hint = f"（本声部 L:{voice_units[vid][0]}/{voice_units[vid][1]} 下对齐至最长小节 {ref} 单位）"
                else:
                    ref = exp_voice
                    if actual == ref:
                        continue
                    hint = f"（本声部 L:{voice_units[vid][0]}/{voice_units[vid][1]}）"

                issues.append(Issue(
                    line_index=v_idx,
                    description=(f"声部 V:{vid} 第 {m_num} 小节时值为 {actual} L单位，"
                                 f"期望 {ref}{hint}"),
                    severity="warning",
                ))

            if auto_fix:
                # 保留原始行结构，逐行修复
                fixed_lines_list = []
                for line_idx, line in enumerate(lines[idxs[0]:idxs[-1]+1], start=idxs[0]):
                    # 对每一行独立处理
                    # 仅在最后一行检查是否需要补休止符（第一行按 first_expected）
                    is_first = (line_idx == idxs[0])
                    is_last = (line_idx == idxs[-1])
                    
                    # 简单修复：对单行应用修复逻辑
                    line_content = lines[line_idx]
                    pre = re.sub(r'"[^"]*"', '',
                          re.sub(r'\[[A-Za-z]:[^\]]*\]', '', line_content))
                    line_measures = [m for m in self._BAR_RE.sub('|', pre).split('|')]
                    
                    # 计算此行小节数和时值
                    line_total = Fraction(0)
                    for measure in line_measures:
                        m_dur = self._measure_duration(measure.strip(), is_cmpd)
                        if m_dur is not None:
                            line_total += m_dur
                    
                    # 对此行应用修复
                    if is_first and max_first_by_vid.get(vid) is not None:
                        fixed_line = self._fix_content(
                            line_content, exp_voice, is_cmpd, max_first_by_vid.get(vid)
                        )
                    else:
                        fixed_line = self._fix_content(
                            line_content, exp_voice, is_cmpd, None
                        )
                    fixed_lines_list.append((line_idx, fixed_line))
                
                # 更新修复后的行
                for line_idx, fixed_line in fixed_lines_list:
                    modified_lines[line_idx] = fixed_line

        if is_global_pickup:
            issues.append(Issue(
                line_index=body_start,
                description="所有声部第一小节均不满足拍号且时值相等，已识别为弱起小节，第一小节未修复",
                severity="info",
            ))

        return issues, modified_lines


class TempoChecker(CheckerModule):
    """
    检查 Q: 速度字段的时值单位与 M: 拍号是否匹配。

    规则：Q: 中明确指定的时值单位分母应与 M: 分母一致。
      - 复合拍号（6/8、9/8、12/8）：标准节拍单位为 3/8（附点四分音符），
        1/8（八分音符）也可接受；Q:1/4 等其他分母视为冲突。
      - 简单拍号（2/4、3/4、4/4 等）：标准节拍单位为 1/分母。

    auto_fix=True 时：将 Q: 时值单位替换为拍号的标准节拍单位，
    并按等效速度重算 BPM（保持实际演奏时间不变）。

    等效 BPM 公式（已知 Q:p/q=bpm，目标单位 P/Q）：
      new_bpm = bpm × (p/q) / (P/Q) = bpm × p × Q / (P × q)
    """

    _COMPOUND = frozenset({(6, 8), (9, 8), (12, 8)})

    def _parse_meter(self, lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in lines:
            s = line.lstrip()
            if not s.startswith('M:'):
                continue
            val = s[2:].strip()
            if val in ('C', 'c'):    return (4, 4)
            if val in ('C|', 'c|'): return (2, 2)
            m = re.match(r'(\d+)/(\d+)', val)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _standard_beat(self, meter: Tuple[int, int]) -> Fraction:
        mn, md = meter
        if (mn, md) in self._COMPOUND:
            return Fraction(3, 8)
        return Fraction(1, md)

    def _parse_q(self, line: str) -> Optional[Tuple[Optional[Fraction], int]]:
        """返回 (beat_unit_or_None, bpm)，或 None 表示不是 Q: 行。"""
        s = line.lstrip()
        if not s.startswith('Q:'):
            return None
        rest = s[2:].strip()
        m = re.match(r'(\d+)/(\d+)\s*=\s*(\d+)', rest)
        if m:
            return Fraction(int(m.group(1)), int(m.group(2))), int(m.group(3))
        m = re.match(r'^(\d+)', rest)
        if m:
            return None, int(m.group(1))
        return None

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        k_idx = next((i for i, l in enumerate(lines) if l.lstrip().startswith('K:')), None)
        if k_idx is None:
            return issues, modified_lines

        meter = self._parse_meter(lines[:k_idx + 1])
        if meter is None:
            return issues, modified_lines

        mn, md = meter
        is_compound = (mn, md) in self._COMPOUND
        std_beat = self._standard_beat(meter)

        for i, line in enumerate(lines[:k_idx + 1]):
            parsed = self._parse_q(line)
            if parsed is None:
                continue
            beat, bpm = parsed
            if beat is None:
                continue

            if beat == std_beat:
                continue
            # 复合拍号下 1/8 也可接受
            if is_compound and beat == Fraction(1, 8):
                continue

            new_bpm = round(Fraction(bpm) * beat / std_beat)
            std_str = f"{std_beat.numerator}/{std_beat.denominator}"
            old_str = f"{beat.numerator}/{beat.denominator}"

            issues.append(Issue(
                line_index=i,
                description=(
                    f"Q: 时值单位 {old_str} 与 M:{mn}/{md} 不匹配"
                    f"（建议改为 Q:{std_str}={new_bpm}，实际演奏速度不变）"
                ),
                severity="warning",
            ))

            if auto_fix:
                stripped = line.lstrip()
                leading = line[:len(line) - len(stripped)]
                rest = stripped[2:]  # "Q:" 之后的全部内容（含标签等）
                new_rest = re.sub(
                    r'\d+/\d+\s*=\s*\d+',
                    f'{std_str}={new_bpm}',
                    rest,
                    count=1,
                )
                modified_lines[i] = f"{leading}Q:{new_rest}"

        return issues, modified_lines


class TempoEstimator(CheckerModule):
    """
    根据 M:、L: 及音符密度估算速度并建议 Q: 字段。

    算法：
      1. 若已存在 Q: 行则直接跳过。
      2. 解析 M: 得到每拍时值（简单拍号取 1/分母；复合拍号 6/8、9/8、12/8 取 3/8）。
      3. 解析 L: 得到基本音符单位。
      4. 扫描乐谱主体，统计所有音符（含单小节休止符 z）时值，
         计算平均时值（L 单位），再换算为每拍平均音符数。
      5. 按密度映射到建议 BPM（每拍音符越多 → 速度越慢，以保证可演奏性）：
           < 1 个/拍  → Allegro  ≈ 132
           1–2 个/拍  → Moderato ≈ 108
           2–4 个/拍  → Andante  ≈ 80
           ≥ 4 个/拍  → Adagio   ≈ 60
      6. 报告建议（severity=info）；auto_fix=True 时在 K: 行前插入 Q: 行。
    """

    _NOTE_DUR_RE = re.compile(
        r'(?:\^\^|__|[\^_=])?'
        r'(?P<pitch>[A-Ga-gz])'
        r"[,']*"
        r'(?P<num>\d*)(?P<slashes>/*)(?P<den>\d*)'
    )

    # (每拍音符数上限, 建议BPM, 速度术语)
    _DENSITY_MAP = [
        (1.0,           132, "Allegro"),
        (2.0,           108, "Moderato"),
        (4.0,            80, "Andante"),
        (float('inf'),   60, "Adagio"),
    ]

    def _parse_meter(self, header_lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in header_lines:
            s = line.lstrip()
            if not s.startswith('M:'):
                continue
            val = s[2:].strip()
            if val in ('C', 'c'):    return (4, 4)
            if val in ('C|', 'c|'): return (2, 2)
            m = re.match(r'(\d+)/(\d+)', val)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _parse_unit(self, header_lines: List[str]) -> Optional[Tuple[int, int]]:
        for line in header_lines:
            s = line.lstrip()
            if not s.startswith('L:'):
                continue
            m = re.match(r'L:\s*(\d+)/(\d+)', s)
            if m:
                return (int(m.group(1)), int(m.group(2)))
        return None

    def _beat_unit(self, meter: Tuple[int, int]) -> Fraction:
        mn, md = meter
        if md == 8 and mn in (6, 9, 12):
            return Fraction(3, 8)
        return Fraction(1, md)

    def _clean_line(self, line: str) -> str:
        line = re.sub(r'%.*$', '', line)
        line = re.sub(r'"[^"]*"', '', line)
        line = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', line)
        line = re.sub(r'\{[^}]*\}', '', line)
        line = re.sub(r'![^!]*!', '', line)
        line = re.sub(r'\+[^+]+\+', '', line)
        line = re.sub(r'[ZX]\d*', '', line)  # 多小节休止符，不计入密度
        return line

    def _collect_durations(self, music_lines: List[str]) -> List[Fraction]:
        durations = []
        for line in music_lines:
            cleaned = self._clean_line(line)
            for m in self._NOTE_DUR_RE.finditer(cleaned):
                num_s   = m.group('num')
                slashes = m.group('slashes')
                den_s   = m.group('den')
                num = int(num_s) if num_s else 1
                if slashes:
                    den = int(den_s) if den_s else 2 ** len(slashes)
                else:
                    den = 1
                durations.append(Fraction(num, den))
        return durations

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        if any(l.lstrip().startswith('Q:') for l in lines):
            return issues, modified_lines

        k_idx = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                k_idx = i
                break
        if k_idx is None:
            return issues, modified_lines

        meter = self._parse_meter(lines[:k_idx + 1])
        if meter is None:
            return issues, modified_lines

        # L: 可能写在 K: 之后的 V: 块内，扩展到全文搜索；
        # 仍找不到时按 ABC 标准从 M: 推导默认值
        unit = self._parse_unit(lines[:k_idx + 1]) or self._parse_unit(lines)
        if unit is None:
            ratio = meter[0] / meter[1]
            unit = (1, 16) if ratio < 0.75 else (1, 8)

        music_lines = [
            l for l in lines[k_idx + 1:]
            if l.strip()
            and not (len(l.lstrip()) >= 2 and l.lstrip()[1] == ':' and l.lstrip()[0].isalpha())
        ]
        durations = self._collect_durations(music_lines)
        if not durations:
            return issues, modified_lines

        avg_dur_l  = sum(durations) / len(durations)
        avg_dur_sb = avg_dur_l * Fraction(unit[0], unit[1])

        beat = self._beat_unit(meter)
        notes_per_beat = float(beat / avg_dur_sb) if avg_dur_sb > 0 else 2.0

        bpm, label = self._DENSITY_MAP[-1][1], self._DENSITY_MAP[-1][2]
        for max_npb, bpm_val, lbl in self._DENSITY_MAP:
            if notes_per_beat <= max_npb:
                bpm, label = bpm_val, lbl
                break

        beat_str = f"{beat.numerator}/{beat.denominator}"
        q_str    = f"Q:{beat_str}={bpm}"

        issues.append(Issue(
            line_index=k_idx,
            description=(
                f"未找到 Q: 速度标记；音符密度约每拍 {notes_per_beat:.1f} 个，"
                f"估算建议速度 {label}，建议添加 {q_str}"
            ),
            severity="info",
        ))

        if auto_fix:
            modified_lines.insert(k_idx, q_str)

        return issues, modified_lines


class BarAccidentalPropagator(CheckerModule):
    """
    检查并修复同小节内临时升降号的传播。

    依据 ABC v2.1 §4.2：^、=、_ 分别表示升号、还原、降号，仅对当前音有效。
    但音频合成器通常不按"小节内传播"解读：某音一旦出现显式升降号，只会渲染该但个音的升降，并不会同小节传播。
    本模块检测缺失的传播标记并在 auto_fix=True 时自动补全，
    使 ABC 乐谱适合直接用于音频生成。

    音高归一化规则（参考 §4.1）：
      大写字母 = 第 0 八度，小写 = 第 1 八度；
      每个 ' 升一个八度，每个 , 降一个八度。
      因此 F' 与 f 属于同一音高，共享升降号状态。
    """

    _BAR_SPLIT_RE = re.compile(r'(:\|:|:\||\|\||\|\]|\[?\|:?)')

    @staticmethod
    def _norm(letter: str, octave_marks: str) -> tuple:
        """(音名大写, 八度整数) 归一化键"""
        pc = letter.upper()
        octave = (1 if letter.islower() else 0) + octave_marks.count("'") - octave_marks.count(',')
        return (pc, octave)

    def _process_segment(self, seg: str, acc_state: dict, auto_fix: bool):
        """
        处理单个小节内容片段（不含小节线）。
        acc_state 原地更新。
        返回 (raw_issues, fixed_seg)；
        raw_issues 为 list of (letter, octave_marks, expected_acc_str)。
        """
        issues = []
        out = []
        i, n = 0, len(seg)

        while i < n:
            c = seg[i]

            # 行注释 → 保留至结尾
            if c == '%':
                out.append(seg[i:]); break

            # 和弦符号 / 注释字符串 "..."
            if c == '"':
                j = i + 1
                while j < n and seg[j] != '"': j += 1
                if j < n: j += 1
                out.append(seg[i:j]); i = j; continue

            # 装饰音 {...} — 内部升降号不影响主音状态
            if c == '{':
                j = i + 1
                while j < n and seg[j] != '}': j += 1
                if j < n: j += 1
                out.append(seg[i:j]); i = j; continue

            # 装饰 !...!
            if c == '!':
                j = i + 1
                while j < n and seg[j] != '!': j += 1
                if j < n: j += 1
                out.append(seg[i:j]); i = j; continue

            # 装饰 +...+
            if c == '+':
                j = i + 1
                while j < n and seg[j] != '+': j += 1
                if j < n: j += 1
                out.append(seg[i:j]); i = j; continue

            # inline 字段 [X:...]
            if c == '[' and i + 2 < n and seg[i+1].isalpha() and seg[i+2] == ':':
                j = i + 1
                while j < n and seg[j] != ']': j += 1
                if j < n: j += 1
                out.append(seg[i:j]); i = j; continue

            # [1 [2 反复结尾标记（不是小节线，不重置）
            if c == '[' and i + 1 < n and seg[i+1].isdigit():
                out.append(c); i += 1; continue

            # 和弦 [notes]
            if c == '[':
                out.append('['); i += 1
                while i < n and seg[i] != ']':
                    if seg[i] in '^_=':
                        acc_str = ''
                        while i < n and seg[i] in '^_=': acc_str += seg[i]; i += 1
                        if i < n and seg[i] in 'ABCDEFGabcdefg':
                            letter = seg[i]; i += 1
                            om = ''
                            while i < n and seg[i] in ",'": om += seg[i]; i += 1
                            acc_state[self._norm(letter, om)] = acc_str
                            out.append(acc_str + letter + om)
                        else:
                            out.append(acc_str)
                    elif seg[i] in 'ABCDEFGabcdefg':
                        letter = seg[i]; i += 1
                        om = ''
                        while i < n and seg[i] in ",'": om += seg[i]; i += 1
                        norm = self._norm(letter, om)
                        if norm in acc_state:
                            exp = acc_state[norm]
                            issues.append((letter, om, exp))
                            out.append((exp if auto_fix else '') + letter + om)
                        else:
                            out.append(letter + om)
                    else:
                        out.append(seg[i]); i += 1
                if i < n: out.append(']'); i += 1
                continue

            # 显式升降号 + 音符
            if c in '^_=':
                acc_str = ''
                while i < n and seg[i] in '^_=': acc_str += seg[i]; i += 1
                if i < n and seg[i] in 'ABCDEFGabcdefg':
                    letter = seg[i]; i += 1
                    om = ''
                    while i < n and seg[i] in ",'": om += seg[i]; i += 1
                    acc_state[self._norm(letter, om)] = acc_str
                    out.append(acc_str + letter + om)
                elif i < n and seg[i] in 'zx':
                    # 升降号后接休止符（罕见，原样保留）
                    out.append(acc_str + seg[i]); i += 1
                else:
                    out.append(acc_str)
                continue

            # 裸音符（无前置升降号）
            if c in 'ABCDEFGabcdefg':
                letter = seg[i]; i += 1
                om = ''
                while i < n and seg[i] in ",'": om += seg[i]; i += 1
                norm = self._norm(letter, om)
                if norm in acc_state:
                    exp = acc_state[norm]
                    issues.append((letter, om, exp))
                    out.append((exp if auto_fix else '') + letter + om)
                else:
                    out.append(letter + om)
                continue

            out.append(c); i += 1

        return issues, ''.join(out)

    def _process_music_line(self, line: str, acc_state: dict, auto_fix: bool):
        """
        处理一行乐谱内容，维护跨行的 acc_state（小节内可跨行）。
        遇到小节线时重置 acc_state。
        返回 (raw_issues, fixed_line, updated_acc_state)。
        """
        parts = self._BAR_SPLIT_RE.split(line)
        result = []
        raw_issues = []

        for j, part in enumerate(parts):
            if j % 2 == 1:
                result.append(part)
                acc_state = {}          # 小节线处重置升降号状态
            else:
                seg_issues, fixed = self._process_segment(part, acc_state, auto_fix)
                raw_issues.extend(seg_issues)
                result.append(fixed)

        return raw_issues, ''.join(result), acc_state

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []
        modified_lines = list(lines)

        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1
                break
        if body_start is None:
            return issues, modified_lines

        def _is_hdr(s: str) -> bool:
            return len(s) >= 2 and s[1] == ':' and s[0].isalpha()

        # 切分声部段
        segments: List[Tuple[Optional[int], List[int]]] = []
        cur_voice: Optional[int] = None
        cur_music: List[int] = []

        for i in range(body_start, len(lines)):
            s = lines[i].lstrip()
            if s.startswith('V:'):
                if cur_music:
                    segments.append((cur_voice, list(cur_music)))
                cur_voice = i
                cur_music = []
            elif s and not _is_hdr(s) and not s.startswith('%'):
                cur_music.append(i)

        if cur_music:
            segments.append((cur_voice, list(cur_music)))

        # 无 V: 时整体作为一段
        if not segments:
            music_idxs = [i for i in range(body_start, len(lines))
                          if lines[i].strip() and not _is_hdr(lines[i].lstrip())
                          and not lines[i].lstrip().startswith('%')]
            if music_idxs:
                segments = [(None, music_idxs)]

        for _, music_idxs in segments:
            acc_state: dict = {}
            for li in music_idxs:
                raw_issues, fixed_line, acc_state = self._process_music_line(
                    lines[li], acc_state, auto_fix
                )
                for (letter, om, exp_acc) in raw_issues:
                    issues.append(Issue(
                        line_index=li,
                        description=(
                            f"小节内音符 {letter}{om} 缺少应延续的升降号，"
                            f"应修正为 {exp_acc}{letter}{om}"
                        ),
                        severity="warning",
                    ))
                if auto_fix:
                    modified_lines[li] = fixed_line

        return issues, modified_lines


class CommentStripper(CheckerModule):
    """
    auto_fix=True 时清理输出：
      - 删除单 % 注释行（保留 %% 指令行）
      - 删除纯空白行（由其他模块合并音乐行后留下的占位行）
    检查模式下不产生任何 issue，直接透传。
    """

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        if not auto_fix:
            return [], list(lines)
        cleaned = []
        for line in lines:
            stripped = line.lstrip()
            if stripped.startswith('%%'):
                cleaned.append(line)
            elif stripped.startswith('%'):
                continue
            elif not stripped:
                continue
            else:
                cleaned.append(line)
        return [], cleaned


class VoiceLineBreakAligner(CheckerModule):
    """
    统一各声部 (V:) 的音乐行换行位置。

    算法：
    1. 按 V: 分段，统计每个声部的音乐行及每行小节数。
    2. 对拥有多行且换行一致（除末行外每行小节数相同）的声部进行投票，
       选出最多声部认可的「小节数/行」作为基准 n（同票时取较小 n，即换行更密）。
    3. 将所有「小节数/行 ≠ n」的声部的音乐行重新按 n 小节/行分行。
    """

    _BAR_SPLIT_RE = re.compile(r'(:\|:|:\||\|\]|\|\||\|:|\|)')

    @staticmethod
    def _is_non_music(s: str) -> bool:
        stripped = s.lstrip()
        if not stripped:
            return True
        if stripped.startswith('%'):
            return True
        if len(stripped) >= 2 and stripped[1] == ':' and stripped[0].isalpha():
            return True
        return False

    def _split_measures(self, content: str) -> List[Tuple[str, str]]:
        """将音乐内容按小节线分割，返回 [(小节内容, 小节线), ...] 列表。"""
        parts = self._BAR_SPLIT_RE.split(content)
        result = []
        i = 0
        while i + 1 < len(parts):
            result.append((parts[i], parts[i + 1]))
            i += 2
        return result

    def _count_measures(self, line: str) -> int:
        return len(self._split_measures(line))

    def _consistent_pattern(self, music_idxs: List[int], lines: List[str]) -> Optional[int]:
        """若各行（末行除外）小节数一致，返回该小节数；否则返回 None。"""
        if len(music_idxs) <= 1:
            return None
        counts = [self._count_measures(lines[i]) for i in music_idxs]
        non_last = counts[:-1]
        if len(set(non_last)) == 1 and non_last[0] > 0:
            return non_last[0]
        return None

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues = []

        body_start = None
        for i, line in enumerate(lines):
            if line.lstrip().startswith('K:'):
                body_start = i + 1
                break
        if body_start is None:
            return issues, list(lines)

        # 按 V: 切分声部段
        segments: List[Dict] = []
        cur_v_idx: Optional[int] = None
        cur_non_music: List[int] = []
        cur_music: List[int] = []
        in_header = True

        def _flush():
            if cur_v_idx is not None:
                segments.append({
                    'v_idx': cur_v_idx,
                    'music_idxs': list(cur_music),
                })

        for i in range(body_start, len(lines)):
            s = lines[i]
            if s.lstrip().startswith('V:'):
                _flush()
                cur_v_idx = i
                cur_non_music = []
                cur_music = []
                in_header = True
            elif cur_v_idx is not None:
                if in_header and self._is_non_music(s):
                    cur_non_music.append(i)
                elif s.strip():
                    in_header = False
                    cur_music.append(i)
        _flush()

        if len(segments) < 2:
            return issues, list(lines)

        # 为每段解析 vid 和换行模式
        for seg in segments:
            vb = lines[seg['v_idx']].lstrip()[2:].strip()
            seg['vid'] = vb.split()[0] if vb else str(seg['v_idx'])
            seg['pattern'] = self._consistent_pattern(seg['music_idxs'], lines)

        # 投票选基准（仅多行且一致的声部参与投票）
        vote_counter: Dict[int, int] = {}
        for seg in segments:
            if seg['pattern'] is not None and seg['music_idxs']:
                vote_counter[seg['pattern']] = vote_counter.get(seg['pattern'], 0) + 1

        if not vote_counter:
            return issues, list(lines)

        # 最多票优先；同票时取较小 n（换行更密）
        canonical = max(vote_counter, key=lambda p: (vote_counter[p], -p))

        to_fix = [
            seg for seg in segments
            if seg['music_idxs'] and seg['pattern'] != canonical
        ]

        if not to_fix:
            return issues, list(lines)

        voice_list = ', '.join(f"V:{seg['vid']}" for seg in to_fix)
        issues.append(Issue(
            line_index=body_start,
            description=(
                f"各声部换行不一致：基准为每 {canonical} 小节换行，"
                f"需对齐的声部：{voice_list}"
            ),
            severity="warning",
        ))

        if not auto_fix:
            return issues, list(lines)

        # 构建替换映射：首个音乐行索引 → 新行列表，其余音乐行 → None（删除）
        replacements: Dict[int, Optional[List[str]]] = {}
        for seg in to_fix:
            music_idxs = seg['music_idxs']
            combined = re.sub(r'\s+', ' ',
                              ' '.join(lines[i].rstrip() for i in music_idxs)).strip()

            raw = self._split_measures(combined)
            if not raw:
                continue

            measures = [(c.strip(), b) for c, b in raw]

            new_lines = []
            for start in range(0, len(measures), canonical):
                chunk = measures[start:start + canonical]
                new_lines.append(' '.join(f'{c} {b}' for c, b in chunk))

            replacements[music_idxs[0]] = new_lines
            for idx in music_idxs[1:]:
                replacements[idx] = None

        # 重建输出行列表
        output: List[str] = []
        for i, line in enumerate(lines):
            if i in replacements:
                repl = replacements[i]
                if repl is not None:
                    output.extend(repl)
                # None → 此行已并入首行，跳过
            else:
                output.append(line)

        return issues, output


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
    sample_abc = """X:1
T:Rhythmic Logic Stress Test
C:Experimental
M:4/4
L:1/8
%%score {V:1 | V:2}
K:C
% --- 右手：包含连音符与超长时值 ---
V:1 nm="Piano" clef=treble
% 第一小节：正常 4/4 (8个1/8音符)
(3cde f2 (5g/a/g/f/e d2 |
% 第二小节：时值溢出 (包含了 10 个 1/8 音符的长度)
e4 (7g/a/b/c'/b/a/g f2 g2 a2 |
% 第三小节：时值缺失 (仅有 6 个 1/8 音符)
[CEG]4 d/e/f/ |
% 第四小节：复杂嵌套感
!trill!g6 (3abc' |]
% --- 左手：与右手时值完全不匹配 ---
V:2 clef=bass
% 第一小节：时值缺失 (仅 4 个 1/8 音符)
C,,2 G,,2 |
% 第二小节：时值正常，但与右手的溢出形成对比
F,,2 A,,2 C,2 E,2 |
% 第三小节：时值大幅溢出 (12 个 1/8 音符)
D,,8 E,,2 F,,2 G,,2 |
% 第四小节：极短小节
C,,1 |]
"""

    engine = ABCProcessor()

    # 规则和默认值通过字典动态配置
    header_rules = {
        "X": "1",
        "T": "Untitled",
        "M": "4/4",
        "L": "1/4",
    }

    engine.register_module(HeaderChecker(required_headers_with_defaults=header_rules))
    engine.register_module(LengthUnifier())
    engine.register_module(TempoChecker())
    engine.register_module(TempoEstimator())
    engine.register_module(VoiceBarCountChecker())
    engine.register_module(MeasureDurationChecker())
    engine.register_module(BarAccidentalPropagator())
    engine.register_module(ClefAutoSelector())
    engine.register_module(VoiceLineBreakAligner())
    engine.register_module(CommentStripper())

    print("执行纯检查模式：")
    issues, _ = engine.run_pipeline(sample_abc, auto_fix=False)
    for issue in issues:
        print(f"[{issue.severity}] 行号 {issue.line_index} -> {issue.description}")

    print("\n执行自动修复模式：")
    issues, fixed_abc = engine.run_pipeline(sample_abc, auto_fix=True)
    print("修复后的乐谱内容：")
    print(fixed_abc)

    # --- L: 统一演示（Czerny 风格：V:1=L:1/8, V:2=L:1/16）---
    print("\n=== L: 统一演示（V:1→L:1/8，V:2→L:1/16）===")
    demo_l = """X:1
T:Carl Czerny
%%score { 1 | 2 }
M:3/8
I:linebreak $
K:G
V:1 treble nm="Piano"
%%MIDI program 0
V:2 bass
%%MIDI program 0
V:1
L:1/8
 d2 g | b2 g | f2 c' | c'af |
V:2
L:1/16
 G,DB,DB,D | G,DB,DB,D | A,DCDCD | A,DCDCD |
"""
    engine3 = ABCProcessor()
    engine3.register_module(LengthUnifier())
    l_issues, l_fixed = engine3.run_pipeline(demo_l, auto_fix=False)
    print("检查结果：")
    for iss in l_issues:
        print(f"  [{iss.severity}] {iss.description}")
    _, l_fixed = engine3.run_pipeline(demo_l, auto_fix=True)
    print("修复后：")
    print(l_fixed)

    # --- Q: 时值一致性检查演示 ---
    print("\n=== Q: 时值检查演示（M:6/8 + Q:1/4=120）===")
    demo_q = """X:1
T:Demo 6/8 Tempo
M:6/8
L:1/8
Q:1/4=120
K:G
def gab|ded BAB|GAB dBG|A3 A3|]
"""
    engine2 = ABCProcessor()
    engine2.register_module(TempoChecker())
    q_issues, q_fixed = engine2.run_pipeline(demo_q, auto_fix=False)
    print("检查结果：")
    for issue in q_issues:
        print(f"  [{issue.severity}] 行号 {issue.line_index} -> {issue.description}")
    _, q_fixed = engine2.run_pipeline(demo_q, auto_fix=True)
    print("修复后：")
    print(q_fixed)

    # --- 临时升降号传播检查演示 ---
    print("\n=== 临时升降号传播检查演示 ===")
    demo_acc = """X:1
T:Accidental Propagation Demo
M:4/4
L:1/4
K:C
% 小节1: ^F 后续 F 未标升号；=B 后续 B 未标还原
^F E F G | B ^c c c |
% 小节2: 降号和双升号传播；小节线后重置
_B A B G | ^^F E F2 |
% 小节3: 和弦内升降号传播至后续裸音
[^FA] F E F |]
"""
    engine_acc = ABCProcessor()
    engine_acc.register_module(BarAccidentalPropagator())
    acc_issues, _ = engine_acc.run_pipeline(demo_acc, auto_fix=False)
    print("检查结果：")
    for iss in acc_issues:
        print(f"  [{iss.severity}] 行号 {iss.line_index} -> {iss.description}")
    _, acc_fixed = engine_acc.run_pipeline(demo_acc, auto_fix=True)
    print("修复后：")
    print(acc_fixed)
