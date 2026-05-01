import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Dict, List, Optional, Tuple


@dataclass
class Issue:
    line_index: int
    description: str
    severity: str


class CheckerModule:
    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        raise NotImplementedError


class HarmonicAnalyzer(CheckerModule):
    """
    分析每个小节的和声属性。

    权重规则（参考 ABC v2.1 §4.1）：
    - 音高越低权重越大（低音决定和声基础）
    - 时值越长权重越大
    - 越靠近小节开头权重越大（开头音更能提示本小节和声）

    算法：
    1. 收集所有声部在同一小节序号的音符（MIDI 音高 + 时值 Fraction）
    2. 以 weight = pitch_weight(midi) × duration × position_weight 累加音级权重
    3. 对 12 个根音 × 10 种和弦类型逐一打分，取最优及次优
    4. 若次优得分 / 最优得分 ≥ 阈值，则对每个声部按时值拆分前/后半小节，
       分别判断和弦，若两半不同则报告和弦变换
    """

    _BAR_SPLIT_RE = re.compile(r'(:\|:|:\||\|\]|\|\||\|:|\|)')

    _BASE_MIDI: Dict[str, int] = {
        'C': 60, 'D': 62, 'E': 64, 'F': 65, 'G': 67, 'A': 69, 'B': 71,
        'c': 72, 'd': 74, 'e': 76, 'f': 77, 'g': 79, 'a': 81, 'b': 83,
    }

    _PC_NAMES = ['C', 'C#', 'D', 'Eb', 'E', 'F', 'F#', 'G', 'Ab', 'A', 'Bb', 'B']
    _PC_NAMES_SHARP = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    _PC_NAMES_FLAT = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

    _CHORD_TYPES: List[Tuple[str, List[int]]] = [
        ('',     [0, 4, 7]),
        ('m',    [0, 3, 7]),
        ('dim',  [0, 3, 6]),
        ('aug',  [0, 4, 8]),
        ('7',    [0, 4, 7, 10]),
        ('maj7', [0, 4, 7, 11]),
        ('m7',   [0, 3, 7, 10]),
        ('dim7', [0, 3, 6, 9]),
        ('sus2', [0, 2, 7]),
        ('sus4', [0, 5, 7]),
    ]

    # 升号/降号加入顺序（音高类别）
    _SHARP_PCS = [5, 0, 7, 2, 9, 4, 11]   # F C G D A E B
    _FLAT_PCS  = [11, 4, 9, 2, 7, 0, 5]   # B E A D G C F

    _KEY_SIG: Dict[str, int] = {
        'C': 0, 'Am': 0,
        'G': 1, 'Em': 1,
        'D': 2, 'Bm': 2,
        'A': 3, 'F#m': 3,
        'E': 4, 'C#m': 4,
        'B': 5, 'G#m': 5,
        'F#': 6, 'D#m': 6,
        'C#': 7, 'A#m': 7,
        'F': -1, 'Dm': -1,
        'Bb': -2, 'Gm': -2,
        'Eb': -3, 'Cm': -3,
        'Ab': -4, 'Fm': -4,
        'Db': -5, 'Bbm': -5,
        'Gb': -6, 'Ebm': -6,
        'Cb': -7, 'Abm': -7,
    }

    _SIG_RELATIVE_KEYS: Dict[int, Tuple[int, int]] = {
        -7: (11, 8),  # Cb / Abm
        -6: (6, 3),   # Gb / Ebm
        -5: (1, 10),  # Db / Bbm
        -4: (8, 5),   # Ab / Fm
        -3: (3, 0),   # Eb / Cm
        -2: (10, 7),  # Bb / Gm
        -1: (5, 2),   # F / Dm
         0: (0, 9),   # C / Am
         1: (7, 4),   # G / Em
         2: (2, 11),  # D / Bm
         3: (9, 6),   # A / F#m
         4: (4, 1),   # E / C#m
         5: (11, 8),  # B / G#m
         6: (6, 3),   # F# / D#m
         7: (1, 10),  # C# / A#m
    }

    _SIG_RELATIVE_NAMES: Dict[int, Tuple[str, str]] = {
        -7: ('Cb', 'Abm'),
        -6: ('Gb', 'Ebm'),
        -5: ('Db', 'Bbm'),
        -4: ('Ab', 'Fm'),
        -3: ('Eb', 'Cm'),
        -2: ('Bb', 'Gm'),
        -1: ('F', 'Dm'),
         0: ('C', 'Am'),
         1: ('G', 'Em'),
         2: ('D', 'Bm'),
         3: ('A', 'F#m'),
         4: ('E', 'C#m'),
         5: ('B', 'G#m'),
         6: ('F#', 'D#m'),
         7: ('C#', 'A#m'),
    }

    # 两个和弦得分之比的阈值：次优 / 最优 ≥ 该值才认为"和弦得分接近"
    _CHANGE_RATIO = 0.85
    # 第二根音 / 第一根音权重之比的阈值：≥ 该值才认为"两个根音都足够大且接近"
    _ROOT_RATIO    = 0.595
    # 每个根音至少要占小节总权重的比例（防止小权重根音凑数）
    _ROOT_MIN_FRAC = 0.15
    # 若七和弦的七音只占该和弦音总权重的很小比例，优先避免过度解释为七和弦
    _WEAK_SEVENTH_MAX_FRAC = 0.15
    _WEAK_SEVENTH_PENALTY = 0.16
    # 根音几乎不存在时，避免把强三和弦误判为弱根音的转位/七和弦
    _WEAK_ROOT_MAX_FRAC = 0.12
    _WEAK_ROOT_PENALTY = 0.12
    # 候选根音明显弱于本小节最强音时，降低转位/替代根音解释的优先级
    _NON_DOMINANT_ROOT_RATIO = 0.75
    _NON_DOMINANT_ROOT_PENALTY = 0.18
    # 最强音作为根音且三和弦完整时，sus 候选常是加音而非真正挂留
    _SUS_OVER_TRIAD_PENALTY = 0.15
    # 候选分数非常接近时，用根音权重决胜
    _ROOT_TIE_MARGIN = 0.05
    # 低音区 MIDI 上限：只有低于该值的音才用于根音竞争判断（中央 C = 60, C5 = 72）
    _BASS_CUTOFF = 72
    # 小节内位置权重：越靠近小节开头，越能代表当前和声基础
    _POSITION_WEIGHT_START = 1.40
    _POSITION_WEIGHT_END = 0.55
    # 功能和声证据接近时，允许强主音/低音中心决定关系大小调。
    _TONIC_CENTER_RATIO = 1.35

    def __init__(self, verbose: bool = False, chord_voice: bool = False):
        self.verbose = verbose
        self.chord_voice = chord_voice

    def _pc_name(self, pc: int, use_sharps: bool) -> str:
        return (self._PC_NAMES_SHARP if use_sharps else self._PC_NAMES_FLAT)[pc % 12]

    def _name_root_pc(self, chord_name: str) -> Optional[int]:
        m = re.match(r'^([A-G][#b]?)(.*)$', chord_name)
        if not m:
            return None
        root = m.group(1)
        pc = {'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11}[root[0]]
        if len(root) > 1:
            pc += 1 if root[1] == '#' else -1
        return pc % 12

    def _name_suffix(self, chord_name: str) -> str:
        m = re.match(r'^[A-G][#b]?(.*)$', chord_name)
        return m.group(1) if m else ''

    def _chord_pcs_for_name(self, chord_name: str) -> set:
        root = self._name_root_pc(chord_name)
        if root is None:
            return set()
        suffix = self._name_suffix(chord_name)
        intervals = next((ivs for suf, ivs in self._CHORD_TYPES if suf == suffix), [0, 4, 7])
        return {(root + iv) % 12 for iv in intervals}

    # ---------- 调号解析 ----------

    def _parse_key_acc(self, lines: List[str]) -> Dict[int, int]:
        """返回 {音高类别: 半音偏移} 的调号升降号映射。"""
        for line in lines:
            s = line.lstrip()
            if not s.startswith('K:'):
                continue
            rest = s[2:].strip()
            key_str = rest.split()[0] if rest else 'C'
            if key_str.lower() in ('none', 'hp'):
                return {}
            n_acc = self._KEY_SIG.get(key_str)
            if n_acc is None:
                m = re.match(r'^([A-G][b#]?)', key_str)
                n_acc = self._KEY_SIG.get(m.group(1), 0) if m else 0
            result: Dict[int, int] = {}
            if n_acc > 0:
                for pc in self._SHARP_PCS[:n_acc]:
                    result[pc] = 1
            elif n_acc < 0:
                for pc in self._FLAT_PCS[:-n_acc]:
                    result[pc] = -1
            return result
        return {}

    def _parse_key_signature_count(self, lines: List[str]) -> int:
        for line in lines:
            s = line.lstrip()
            if not s.startswith('K:'):
                continue
            rest = s[2:].strip()
            key_str = rest.split()[0] if rest else 'C'
            if key_str.lower() in ('none', 'hp'):
                return 0
            n_acc = self._KEY_SIG.get(key_str)
            if n_acc is not None:
                return n_acc
            m = re.match(r'^([A-G][b#]?)', key_str)
            return self._KEY_SIG.get(m.group(1), 0) if m else 0
        return 0

    def _line_ending(self, line: str, fallback: str = '') -> str:
        if line.endswith('\r\n'):
            return '\r\n'
        if line.endswith('\n'):
            return '\n'
        return fallback

    def _replace_key_field_key(self, line: str, new_key: str) -> str:
        """
        更新 K: 行的调名，同时保留谱号、transpose 等附加参数。
        如果 K: 行只有 clef shorthand（如 K:bass），则插入调名而不是覆盖谱号。
        """
        nl = self._line_ending(line)
        body = line[:-len(nl)] if nl else line
        m = re.match(r'(\s*K:\s*)(\S+)?(.*)$', body)
        if not m:
            return line

        prefix, key_token, tail = m.group(1), m.group(2), m.group(3)
        if not key_token:
            return f'{prefix}{new_key}{tail}{nl}'

        token_l = key_token.lower()
        key_like = (
            token_l in ('none', 'hp')
            or key_token in self._KEY_SIG
            or re.match(r'^[A-G][b#]?(?:m|min|minor|maj|major|ion|dor|phr|lyd|mix|aeo|loc)?$',
                        key_token, re.IGNORECASE)
        )
        if key_like:
            return f'{prefix}{new_key}{tail}{nl}'
        return f'{prefix}{new_key} {key_token}{tail}{nl}'

    def _ensure_voice_key_fields(self, lines: List[str], k_idx: int, key_name: str) -> List[str]:
        """在全局 K: 之后的每个 V: 行后插入/更新对应声部的 K: 行。"""
        result: List[str] = []
        fallback_nl = next((self._line_ending(l) for l in lines if self._line_ending(l)), '')
        i = 0
        while i < len(lines):
            line = lines[i]
            result.append(line)

            if i > k_idx and line.lstrip().startswith('V:'):
                next_i = i + 1
                indent = re.match(r'\s*', line).group(0)
                nl = self._line_ending(line, fallback_nl)
                if next_i < len(lines) and lines[next_i].lstrip().startswith('K:'):
                    result.append(self._replace_key_field_key(lines[next_i], key_name))
                    i += 2
                    continue
                result.append(f'{indent}K:{key_name}{nl}')

            i += 1
        return result

    def _mode_bias_for_weights(self, pcw: Dict[int, float], sig_count: int) -> str:
        """
        在同一调号下比较关系大调/小调的和弦证据。
        返回 'major'、'minor' 或 'neutral'；小调侧包含和声小调 V / vii° 的升七级。
        """
        if not pcw:
            return 'neutral'
        major_tonic, minor_tonic = self._SIG_RELATIVE_KEYS.get(sig_count, (0, 9))

        def triad_weight(root: int, third: int, fifth: int) -> float:
            rw = pcw.get(root, 0.0)
            tw = pcw.get((root + third) % 12, 0.0)
            fw = pcw.get((root + fifth) % 12, 0.0)
            if rw <= 0 or (tw <= 0 and fw <= 0):
                return 0.0
            return rw + 0.85 * tw + 0.65 * fw

        major_degrees = [
            (0, 4, 7), (2, 3, 7), (4, 3, 7),
            (5, 4, 7), (7, 4, 7), (9, 3, 7), (11, 3, 6),
        ]
        minor_degrees = [
            (0, 3, 7), (2, 3, 6), (3, 4, 7),
            (5, 3, 7), (7, 3, 7), (8, 4, 7), (10, 4, 7),
            (7, 4, 7), (11, 3, 6),  # harmonic minor: V, vii°
        ]
        major_score = sum(
            triad_weight((major_tonic + deg) % 12, third, fifth)
            for deg, third, fifth in major_degrees
        )
        minor_score = sum(
            triad_weight((minor_tonic + deg) % 12, third, fifth)
            for deg, third, fifth in minor_degrees
        )
        if minor_score > major_score * 1.08:
            return 'minor'
        if major_score > minor_score * 1.08:
            return 'major'
        return 'neutral'

    def _infer_relative_key(self, notes: List[Tuple[Fraction, int, Fraction]],
                            sig_count: int) -> Tuple[str, str, float]:
        major_name, minor_name = self._SIG_RELATIVE_NAMES.get(sig_count, ('C', 'Am'))
        if not notes:
            return major_name, 'major', 1.0

        pcw = self._timed_pc_weights(notes)
        bass_pcw = self._bass_timed_pc_weights(notes)
        major_tonic, minor_tonic = self._SIG_RELATIVE_KEYS.get(sig_count, (0, 9))
        all_mode = self._mode_bias_for_weights(pcw, sig_count)
        bass_mode = self._mode_bias_for_weights(bass_pcw, sig_count)

        major_w = pcw.get(major_tonic, 0.0) + 1.2 * bass_pcw.get(major_tonic, 0.0)
        minor_w = pcw.get(minor_tonic, 0.0) + 1.2 * bass_pcw.get(minor_tonic, 0.0)
        confidence = (max(major_w, minor_w) + 1e-9) / (min(major_w, minor_w) + 1e-9)

        if all_mode == 'minor' and bass_mode in ('minor', 'neutral') and minor_w >= major_w * 1.08:
            return minor_name, 'minor', confidence
        if all_mode == 'major' and bass_mode in ('major', 'neutral') and major_w >= minor_w * 1.08:
            return major_name, 'major', confidence
        if all_mode == 'neutral' and bass_mode == 'neutral':
            if minor_w >= major_w * self._TONIC_CENTER_RATIO:
                return minor_name, 'minor', confidence
            if major_w >= minor_w * self._TONIC_CENTER_RATIO:
                return major_name, 'major', confidence
        return major_name, 'neutral', confidence

    def _diatonic_third_suffix(self, root: int, key_acc: Dict[int, int],
                               mode_bias: str = 'neutral',
                               sig_count: Optional[int] = None) -> str:
        """根据当前调号判断 root 上方调内三度是大三度还是小三度。"""
        if sig_count is not None and mode_bias in ('major', 'minor'):
            major_tonic, minor_tonic = self._SIG_RELATIVE_KEYS.get(sig_count, (0, 9))
            tonic = minor_tonic if mode_bias == 'minor' else major_tonic
            # 自然小调里 v 默认是小三和弦；只有本小节实际出现导音时，
            # 后面的功能和声修正才会把它提升为大属/属七。
            minor_degrees = {0, 2, 5, 7}
            major_degrees = {3, 8, 10}
            for deg in range(12):
                if (tonic + deg) % 12 != root:
                    continue
                if mode_bias == 'minor':
                    if deg in minor_degrees:
                        return 'm'
                    if deg in major_degrees:
                        return ''
                else:
                    if deg in {2, 4, 9}:
                        return 'm'
                    if deg in {0, 5, 7}:
                        return ''

        for ltr, natural_pc in self._NATURAL_BASE:
            if (natural_pc + key_acc.get(natural_pc, 0)) % 12 != root:
                continue
            idx = next(i for i, (n_ltr, _) in enumerate(self._NATURAL_BASE) if n_ltr == ltr)
            third_natural = self._NATURAL_BASE[(idx + 2) % len(self._NATURAL_BASE)][1]
            third_pc = (third_natural + key_acc.get(third_natural, 0)) % 12
            interval = (third_pc - root) % 12
            if interval == 3:
                return 'm'
            if interval == 4:
                return ''
        return ''

    # ---------- 音符解析 ----------

    def _parse_dur(self, s: str, pos: int) -> Tuple[Fraction, int]:
        n = len(s)
        num_s = ''
        while pos < n and s[pos].isdigit(): num_s += s[pos]; pos += 1
        slashes = 0
        while pos < n and s[pos] == '/': slashes += 1; pos += 1
        den_s = ''
        while pos < n and s[pos].isdigit(): den_s += s[pos]; pos += 1
        if not num_s and not slashes:
            return Fraction(1), pos
        num = int(num_s) if num_s else 1
        den = (int(den_s) if den_s else 2 ** slashes) if slashes else 1
        return Fraction(num, den), pos

    def _to_midi(self, letter: str, octave_marks: str,
                 acc_str: str, key_acc: Dict[int, int]) -> int:
        midi = self._BASE_MIDI[letter]
        midi += 12 * octave_marks.count("'")
        midi -= 12 * octave_marks.count(',')
        if acc_str:
            delta = 0 if '=' in acc_str else acc_str.count('^') - acc_str.count('_')
            midi += delta
        else:
            midi += key_acc.get(midi % 12, 0)
        return midi

    def _extract_notes(self, seg: str,
                       key_acc: Dict[int, int]) -> List[Tuple[int, Fraction]]:
        """从一个小节片段中提取 (midi, duration) 列表。"""
        s = re.sub(r'%.*$', '', seg, flags=re.MULTILINE)
        s = re.sub(r'"[^"]*"', '', s)
        s = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', s)
        s = re.sub(r'\{[^}]*\}', '', s)
        s = re.sub(r'![^!]*!', '', s)
        s = re.sub(r'\+[^+]+\+', '', s)

        notes: List[Tuple[int, Fraction]] = []
        i, n = 0, len(s)
        tuplet_stack: List[List] = []
        _TQ = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}

        def add(midi: int, dur: Fraction) -> None:
            actual = dur
            if tuplet_stack:
                actual = dur * tuplet_stack[-1][1]
                tuplet_stack[-1][0] -= 1
                if tuplet_stack[-1][0] <= 0:
                    tuplet_stack.pop()
            notes.append((midi, actual))

        while i < n:
            c = s[i]
            if c in ' \t\n~HLMOPSTuv.-_&\\': i += 1; continue
            if c in '<>': i += 1; continue

            if c == '(':
                i += 1
                p_s = ''
                while i < n and s[i].isdigit(): p_s += s[i]; i += 1
                if not p_s: continue
                p = int(p_s); q = _TQ.get(p, 2); r = p
                if i < n and s[i] == ':':
                    i += 1
                    q_s = ''
                    while i < n and s[i].isdigit(): q_s += s[i]; i += 1
                    if q_s: q = int(q_s)
                    if i < n and s[i] == ':':
                        i += 1
                        r_s = ''
                        while i < n and s[i].isdigit(): r_s += s[i]; i += 1
                        if r_s: r = int(r_s)
                if p > 0:
                    tuplet_stack.append([r, Fraction(q, p)])
                continue

            if c == '[':
                i += 1
                if i < n and s[i].isdigit(): i += 1; continue
                chord_notes: List[Tuple[int, Fraction]] = []
                while i < n and s[i] != ']':
                    acc = ''
                    while i < n and s[i] in '^_=': acc += s[i]; i += 1
                    if i < n and s[i] in 'ABCDEFGabcdefg':
                        letter = s[i]; i += 1
                        om = ''
                        while i < n and s[i] in ",'": om += s[i]; i += 1
                        inner_dur, i = self._parse_dur(s, i)
                        chord_notes.append((
                            self._to_midi(letter, om, acc, key_acc),
                            inner_dur,
                        ))
                    elif i < n and s[i] in 'zx':
                        i += 1
                        while i < n and (s[i].isdigit() or s[i] == '/'): i += 1
                    else:
                        i += 1
                if i < n and s[i] == ']': i += 1
                outer_dur, i = self._parse_dur(s, i)
                actual_outer = outer_dur
                if tuplet_stack:
                    actual_outer = outer_dur * tuplet_stack[-1][1]
                    tuplet_stack[-1][0] -= 1
                    if tuplet_stack[-1][0] <= 0:
                        tuplet_stack.pop()
                for m, inner_dur in chord_notes:
                    notes.append((m, inner_dur * actual_outer))
                continue

            if c in '^_=':
                acc = ''
                while i < n and s[i] in '^_=': acc += s[i]; i += 1
                if i < n and s[i] in 'ABCDEFGabcdefg':
                    letter = s[i]; i += 1
                    om = ''
                    while i < n and s[i] in ",'": om += s[i]; i += 1
                    dur, i = self._parse_dur(s, i)
                    add(self._to_midi(letter, om, acc, key_acc), dur)
                continue

            if c in 'ABCDEFGabcdefg':
                letter = c; i += 1
                om = ''
                while i < n and s[i] in ",'": om += s[i]; i += 1
                dur, i = self._parse_dur(s, i)
                add(self._to_midi(letter, om, '', key_acc), dur)
                continue

            if c in 'zxZX':
                i += 1
                while i < n and s[i].isdigit(): i += 1
                continue

            i += 1

        return notes

    # ---------- 和弦声部生成辅助 ----------

    _NATURAL_BASE = [('C',0),('D',2),('E',4),('F',5),('G',7),('A',9),('B',11)]

    def _pc_to_abc_in_key(self, pc: int, key_acc: Dict[int, int], use_sharps: bool) -> str:
        """返回在当前调号下发出指定音级的 ABC 音名前缀（不含八度符号）。"""
        # 1. 调号内的自然音名（无需额外升降号）
        for ltr, bp in self._NATURAL_BASE:
            if (bp + key_acc.get(bp, 0)) % 12 == pc:
                return ltr
        # 2. 需要升 / 降号。升号调优先使用升号拼写，避免 A#m 被写成 [^A,C=F]。
        if use_sharps:
            for ltr, bp in self._NATURAL_BASE:
                if (bp + 1) % 12 == pc:
                    return '^' + ltr
            for ltr, bp in self._NATURAL_BASE:
                if bp % 12 == pc and key_acc.get(bp, 0) != 0:
                    return '=' + ltr
        else:
            for ltr, bp in self._NATURAL_BASE:
                if bp % 12 == pc and key_acc.get(bp, 0) != 0:
                    return '=' + ltr
            for ltr, bp in self._NATURAL_BASE:
                if (bp - 1 + 12) % 12 == pc:
                    return '_' + ltr
        # 4. 回退
        for ltr, bp in self._NATURAL_BASE:
            if (bp + 1) % 12 == pc:
                return '^' + ltr
        return '?'

    def _midi_to_abc_str(self, midi: int, key_acc: Dict[int, int], use_sharps: bool) -> str:
        """将 MIDI 值转换为完整 ABC 音名（含八度符号，不含时值）。"""
        pc = midi % 12
        note_name = self._pc_to_abc_in_key(pc, key_acc, use_sharps)
        abc_oct = midi // 12 - 5   # 0 = C4-B4 range; -1 = C3-B3; 1 = C5-B5
        if abc_oct >= 1:
            converted = ''.join(c.lower() if c.isalpha() else c for c in note_name)
            return converted + "'" * (abc_oct - 1)
        elif abc_oct <= -1:
            return note_name + ',' * (-abc_oct)
        return note_name

    def _dur_to_str(self, dur: Fraction) -> str:
        """将 Fraction 时值转换为 ABC 时值字符串。"""
        if dur == 1:
            return ''
        if dur.denominator == 1:
            return str(dur.numerator)
        if dur.numerator == 1:
            return f'/{dur.denominator}'
        return f'{dur.numerator}/{dur.denominator}'

    def _chord_name_to_abc(self, chord_name: str, dur: Fraction,
                            key_acc: Dict[int, int], use_sharps: bool,
                            inversion_pc: Optional[int] = None,
                            drop_root: bool = False,
                            drop_fifth: bool = False) -> str:
        """将和弦名（如 'F#m'、'Emaj7'）转换为 ABC 柱式和弦，根音置于第 3 八度。"""
        pm = re.match(r'^([A-G][#b]?)(.*)$', chord_name)
        if not pm:
            return 'z' + self._dur_to_str(dur)
        root_str, suffix = pm.group(1), pm.group(2)

        base_pc = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[root_str[0]]
        if len(root_str) > 1:
            base_pc = (base_pc + (1 if root_str[1] == '#' else -1)) % 12

        ivs = [0, 4, 7]
        for suf, intervals in self._CHORD_TYPES:
            if suf == suffix:
                ivs = intervals
                break

        chord_pcs = [(base_pc + iv) % 12 for iv in ivs]
        if inversion_pc in chord_pcs:
            start = chord_pcs.index(inversion_pc)
            chord_pcs = chord_pcs[start:] + chord_pcs[:start]

        root_midi = 48 + chord_pcs[0]  # 第 3 八度：C3=48 … B3=59
        midi_notes = [root_midi]
        for tgt_pc in chord_pcs[1:]:
            prev = midi_notes[-1]
            cand = (prev // 12) * 12 + tgt_pc
            if cand <= prev:
                cand += 12
            midi_notes.append(cand)
        if drop_root and len(midi_notes) >= 3 and chord_pcs[-1] == base_pc:
            midi_notes[-1] = 48 + base_pc
        if drop_fifth and len(midi_notes) >= 3 and chord_pcs[-1] == (base_pc + 7) % 12:
            for idx, pc in enumerate(chord_pcs[1:], start=1):
                midi_notes[idx] = 48 + pc

        notes_str = ''.join(self._midi_to_abc_str(m, key_acc, use_sharps) for m in midi_notes)
        return f'[{notes_str}]{self._dur_to_str(dur)}'

    # ---------- 和声评分 ----------

    def _pitch_weight(self, midi: int) -> float:
        """音越低权重越大：中央 C (60) = 1.0，每降一个八度 +1.0，上限 0.5。"""
        return max(0.5, 1.0 + (60 - midi) / 12.0)

    def _pc_weights(self, notes: List[Tuple[int, Fraction]]) -> Dict[int, float]:
        w: Dict[int, float] = {}
        for midi, dur in notes:
            pc = midi % 12
            weight = self._pitch_weight(midi) * float(dur)
            w[pc] = w.get(pc, 0.0) + weight
        return w

    def _position_weight(self, start: Fraction, dur: Fraction,
                         total_span: Fraction) -> float:
        if total_span <= 0:
            return 1.0

        midpoint = float(start + dur / 2) / float(total_span)
        midpoint = max(0.0, min(1.0, midpoint))
        return (
            self._POSITION_WEIGHT_START
            - (self._POSITION_WEIGHT_START - self._POSITION_WEIGHT_END) * midpoint
        )

    def _timed_pc_weights(
        self,
        notes: List[Tuple[Fraction, int, Fraction]],
    ) -> Dict[int, float]:
        if not notes:
            return {}

        base_t = min(t for t, _, _ in notes)
        total_end = max(t + dur for t, _, dur in notes)
        total_span = total_end - base_t

        w: Dict[int, float] = {}
        for t, midi, dur in notes:
            pc = midi % 12
            rel_t = t - base_t
            weight = (
                self._pitch_weight(midi)
                * float(dur)
                * self._position_weight(rel_t, dur, total_span)
            )
            w[pc] = w.get(pc, 0.0) + weight
        return w

    def _bass_timed_pc_weights(
        self,
        notes: List[Tuple[Fraction, int, Fraction]],
    ) -> Dict[int, float]:
        """只统计低音区（MIDI < _BASS_CUTOFF）音符的音级权重，用于根音竞争判断。"""
        bass_notes = [(t, midi, dur) for t, midi, dur in notes if midi < self._BASS_CUTOFF]
        return self._timed_pc_weights(bass_notes)

    def _score_chord(self, pcw: Dict[int, float],
                     root: int, intervals: List[int]) -> float:
        chord_pcs = {(root + iv) % 12 for iv in intervals}
        total = sum(pcw.values())
        if total == 0:
            return 0.0
        in_chord = sum(pcw.get(pc, 0.0) for pc in chord_pcs)
        coverage = in_chord / total
        penalty = (total - in_chord) / total
        # 和弦完备率：和弦音中实际出现（权重>0）的比例；
        # 缺席的和弦音说明该和弦对这组音而言可能是过度解释。
        present = sum(1 for pc in chord_pcs if pcw.get(pc, 0.0) > 0)
        completeness = present / len(chord_pcs)
        return coverage - 0.5 * penalty + 0.1 * completeness

    def _adjust_chord_score(self, pcw: Dict[int, float],
                            root: int, suffix: str,
                            intervals: List[int], score: float) -> float:
        chord_pcs = [(root + iv) % 12 for iv in intervals]
        chord_w = sum(pcw.get(pc, 0.0) for pc in chord_pcs)
        if chord_w == 0:
            return score

        strongest_pc, strongest_w = max(pcw.items(), key=lambda x: x[1])
        root_w = pcw.get(root, 0.0)
        # 只有最强音不属于候选和弦时才惩罚弱根音；若最强音是和弦内音（五度、三度等），
        # 则根音弱于最强音属于正常情况，不应降低评分。
        if root != strongest_pc and strongest_w > 0 and strongest_pc not in chord_pcs:
            if root_w / strongest_w < self._NON_DOMINANT_ROOT_RATIO:
                score -= self._NON_DOMINANT_ROOT_PENALTY

        if root == strongest_pc and suffix in ('sus2', 'sus4'):
            major = {(root + iv) % 12 for iv in (0, 4, 7)}
            minor = {(root + iv) % 12 for iv in (0, 3, 7)}
            if (
                all(pcw.get(pc, 0.0) > 0 for pc in major)
                or all(pcw.get(pc, 0.0) > 0 for pc in minor)
            ):
                score -= self._SUS_OVER_TRIAD_PENALTY

        root_frac = pcw.get(root, 0.0) / chord_w
        if root_frac < self._WEAK_ROOT_MAX_FRAC:
            score -= self._WEAK_ROOT_PENALTY

        if suffix not in ('7', 'maj7', 'm7', 'dim7') or len(intervals) < 4:
            return score

        seventh_pc = chord_pcs[3]
        seventh_frac = pcw.get(seventh_pc, 0.0) / chord_w
        if seventh_frac < self._WEAK_SEVENTH_MAX_FRAC:
            return score - self._WEAK_SEVENTH_PENALTY

        return score

    def _top_chords(self, pcw: Dict[int, float],
                    top_n: int = 2,
                    use_sharps: bool = False) -> List[Tuple[str, float]]:
        if not pcw:
            return []
        candidates = []
        for root in range(12):
            for suffix, ivs in self._CHORD_TYPES:
                score = self._score_chord(pcw, root, ivs)
                if score > 0:
                    score = self._adjust_chord_score(pcw, root, suffix, ivs, score)
                    root_w = pcw.get(root, 0.0)
                    candidates.append((self._pc_name(root, use_sharps) + suffix, score, root_w))
        candidates.sort(key=lambda x: (-x[1], -x[2]))

        best_score = candidates[0][1]
        close = [c for c in candidates if best_score - c[1] <= self._ROOT_TIE_MARGIN]
        rest = [c for c in candidates if best_score - c[1] > self._ROOT_TIE_MARGIN]
        close.sort(key=lambda x: (-x[2], -x[1]))

        ranked = close + rest
        return [(name, score) for name, score, _ in ranked[:top_n]]

    def _best_chord_for_root(self, pcw: Dict[int, float], root: int,
                             key_acc: Optional[Dict[int, int]] = None,
                             mode_bias: str = 'neutral',
                             sig_count: Optional[int] = None,
                             use_sharps: bool = False) -> Optional[str]:
        """为已知根音选和弦名；优先避免把明确根音段落解释成倒置和弦。"""
        if not pcw:
            return None

        root_w = pcw.get(root, 0.0)
        if root_w > 0:
            major_third = (root + 4) % 12
            minor_third = (root + 3) % 12
            fifth = (root + 7) % 12
            major_w = pcw.get(major_third, 0.0)
            minor_w = pcw.get(minor_third, 0.0)
            fifth_w = pcw.get(fifth, 0.0)

            if minor_w > 0 and major_w == 0:
                return self._pc_name(root, use_sharps) + 'm'
            if major_w > 0 and minor_w == 0:
                return self._pc_name(root, use_sharps)

            # 根音明确但三音缺席时，使用调号内三度作为保守命名；
            # 避免 C-G-Bb 这类无三音材料被过度解释成 C7。
            if key_acc is not None and (fifth_w > 0 or len(pcw) == 1):
                return (
                    self._pc_name(root, use_sharps)
                    + self._diatonic_third_suffix(root, key_acc, mode_bias, sig_count)
                )

        complete_simple: List[Tuple[str, float, float]] = []
        for suffix, ivs in self._CHORD_TYPES:
            if suffix not in ('', 'm', 'sus2', 'sus4'):
                continue
            chord_pcs = {(root + iv) % 12 for iv in ivs}
            if all(pcw.get(pc, 0.0) > 0 for pc in chord_pcs):
                score = self._score_chord(pcw, root, ivs)
                in_chord = sum(pcw.get(pc, 0.0) for pc in chord_pcs)
                complete_simple.append((self._pc_name(root, use_sharps) + suffix, score, in_chord))

        if complete_simple:
            complete_simple.sort(key=lambda x: (-x[2], -x[1]))
            return complete_simple[0][0]

        return None

    def _prefer_root_chord(self, pcw: Dict[int, float], root: int,
                           current_name: str, current_score: float,
                           key_acc: Dict[int, int], mode_bias: str,
                           sig_count: int, use_sharps: bool) -> Optional[str]:
        root_name = self._best_chord_for_root(
            pcw, root, key_acc, mode_bias, sig_count, use_sharps
        )
        if not root_name or root_name == current_name:
            return None

        root_w = pcw.get(root, 0.0)
        strongest_w = max(pcw.values()) if pcw else 0.0
        if strongest_w <= 0 or root_w / strongest_w < 0.92:
            return None

        current_root = self._name_root_pc(current_name)
        if current_root is not None and (root - current_root) % 12 == 7:
            return None

        suffix = self._name_suffix(root_name)
        intervals = next((ivs for suf, ivs in self._CHORD_TYPES if suf == suffix), [0, 4, 7])
        root_score = self._adjust_chord_score(
            pcw, root, suffix, intervals, self._score_chord(pcw, root, intervals)
        )
        if root_score >= current_score * 0.68:
            return root_name
        return None

    def _prefer_complete_inversion(self, pcw: Dict[int, float], current_name: str,
                                   current_score: float, use_sharps: bool) -> Optional[str]:
        if not pcw:
            return None
        current_root = self._name_root_pc(current_name)
        candidates: List[Tuple[str, float, float, float]] = []
        current_pcs = self._chord_pcs_for_name(current_name)
        current_in_chord = sum(pcw.get(pc, 0.0) for pc in current_pcs)
        for root in range(12):
            for suffix, ivs in self._CHORD_TYPES:
                if suffix not in ('', 'm', 'dim'):
                    continue
                chord_pcs = {(root + iv) % 12 for iv in ivs}
                if not all(pcw.get(pc, 0.0) > 0 for pc in chord_pcs):
                    continue
                in_chord = sum(pcw.get(pc, 0.0) for pc in chord_pcs)
                score = self._adjust_chord_score(
                    pcw, root, suffix, ivs, self._score_chord(pcw, root, ivs)
                )
                root_w = pcw.get(root, 0.0)
                candidates.append((self._pc_name(root, use_sharps) + suffix, score, in_chord, root_w))
        if not candidates:
            return None
        candidates.sort(key=lambda x: (-x[2], -x[1], -x[3]))
        name, score, in_chord, _ = candidates[0]
        root = self._name_root_pc(name)
        if (
            name != current_name
            and root is not None
            and root != current_root
            and in_chord >= current_in_chord * 1.08
            and score >= current_score * 0.70
        ):
            return name
        return None

    def _simplify_close_seventh(self, pcw: Dict[int, float], chord_name: str,
                                score: float, use_sharps: bool) -> str:
        root = self._name_root_pc(chord_name)
        suffix = self._name_suffix(chord_name)
        if root is None or suffix not in ('7', 'm7', 'maj7'):
            return chord_name
        simple_suffix = 'm' if suffix == 'm7' else ''
        simple_ivs = next((ivs for suf, ivs in self._CHORD_TYPES if suf == simple_suffix), [0, 4, 7])
        simple_score = self._adjust_chord_score(
            pcw, root, simple_suffix, simple_ivs, self._score_chord(pcw, root, simple_ivs)
        )
        if simple_score >= score * 0.94:
            return self._pc_name(root, use_sharps) + simple_suffix
        return chord_name

    def _named_score(self, pcw: Dict[int, float], chord_name: str) -> float:
        root = self._name_root_pc(chord_name)
        if root is None:
            return 0.0
        suffix = self._name_suffix(chord_name)
        intervals = next((ivs for suf, ivs in self._CHORD_TYPES if suf == suffix), [0, 4, 7])
        return self._adjust_chord_score(
            pcw, root, suffix, intervals, self._score_chord(pcw, root, intervals)
        )

    def _chord_weight(self, pcw: Dict[int, float], chord_name: str) -> float:
        return sum(pcw.get(pc, 0.0) for pc in self._chord_pcs_for_name(chord_name))

    def _has_all_pcs(self, pcw: Dict[int, float], pcs: List[int]) -> bool:
        return all(pcw.get(pc % 12, 0.0) > 0 for pc in pcs)

    # ---------- 小节分割 ----------

    def _split_measures(self, music: str) -> List[str]:
        parts = self._BAR_SPLIT_RE.split(music)
        return parts[::2]

    # ---------- 带时间戳的音符提取（用于跨声部对齐） ----------

    def _extract_timed_notes(self, seg: str,
                             key_acc: Dict[int, int]) -> List[Tuple[Fraction, int, Fraction]]:
        """
        提取 (start_time, midi, duration) 列表，同一和弦内的音符共享相同 start_time。
        休止符 (z/x/Z/X) 不产生条目，但会推进时间轴。
        """
        s = re.sub(r'%.*$', '', seg, flags=re.MULTILINE)
        s = re.sub(r'"[^"]*"', '', s)
        s = re.sub(r'\[[A-Za-z]:[^\]]*\]', '', s)
        s = re.sub(r'\{[^}]*\}', '', s)
        s = re.sub(r'![^!]*!', '', s)
        s = re.sub(r'\+[^+]+\+', '', s)

        result: List[Tuple[Fraction, int, Fraction]] = []
        t = Fraction(0)
        i, n = 0, len(s)
        tuplet_stack: List[List] = []
        _TQ = {2: 3, 3: 2, 4: 3, 5: 2, 6: 2, 7: 2, 8: 3, 9: 2}

        def actual_dur(raw: Fraction) -> Fraction:
            d = raw
            if tuplet_stack:
                d = raw * tuplet_stack[-1][1]
                tuplet_stack[-1][0] -= 1
                if tuplet_stack[-1][0] <= 0:
                    tuplet_stack.pop()
            return d

        while i < n:
            c = s[i]
            if c in ' \t\n~HLMOPSTuv.-_&\\': i += 1; continue
            if c in '<>': i += 1; continue

            if c == '(':
                i += 1
                p_s = ''
                while i < n and s[i].isdigit(): p_s += s[i]; i += 1
                if not p_s: continue
                p = int(p_s); q = _TQ.get(p, 2); r = p
                if i < n and s[i] == ':':
                    i += 1
                    q_s = ''
                    while i < n and s[i].isdigit(): q_s += s[i]; i += 1
                    if q_s: q = int(q_s)
                    if i < n and s[i] == ':':
                        i += 1
                        r_s = ''
                        while i < n and s[i].isdigit(): r_s += s[i]; i += 1
                        if r_s: r = int(r_s)
                if p > 0:
                    tuplet_stack.append([r, Fraction(q, p)])
                continue

            if c == '[':
                i += 1
                if i < n and s[i].isdigit(): i += 1; continue
                chord_notes: List[Tuple[int, Fraction]] = []
                while i < n and s[i] != ']':
                    acc = ''
                    while i < n and s[i] in '^_=': acc += s[i]; i += 1
                    if i < n and s[i] in 'ABCDEFGabcdefg':
                        letter = s[i]; i += 1
                        om = ''
                        while i < n and s[i] in ",'": om += s[i]; i += 1
                        inner_dur, i = self._parse_dur(s, i)
                        chord_notes.append((
                            self._to_midi(letter, om, acc, key_acc),
                            inner_dur,
                        ))
                    elif i < n and s[i] in 'zx':
                        i += 1
                        while i < n and (s[i].isdigit() or s[i] == '/'): i += 1
                    else:
                        i += 1
                if i < n and s[i] == ']': i += 1
                outer_raw, i = self._parse_dur(s, i)
                outer_dur = actual_dur(outer_raw)
                chord_durs = [inner * outer_dur for _, inner in chord_notes]
                for (m, _), d in zip(chord_notes, chord_durs):
                    result.append((t, m, d))   # 同一和弦共享 start_time
                if chord_durs:
                    t += max(chord_durs)        # 时间轴按最长和弦音推进一次
                continue

            if c in '^_=':
                acc = ''
                while i < n and s[i] in '^_=': acc += s[i]; i += 1
                if i < n and s[i] in 'ABCDEFGabcdefg':
                    letter = s[i]; i += 1
                    om = ''
                    while i < n and s[i] in ",'": om += s[i]; i += 1
                    raw, i = self._parse_dur(s, i)
                    d = actual_dur(raw)
                    result.append((t, self._to_midi(letter, om, acc, key_acc), d))
                    t += d
                continue

            if c in 'ABCDEFGabcdefg':
                letter = c; i += 1
                om = ''
                while i < n and s[i] in ",'": om += s[i]; i += 1
                raw, i = self._parse_dur(s, i)
                d = actual_dur(raw)
                result.append((t, self._to_midi(letter, om, '', key_acc), d))
                t += d
                continue

            if c in 'zx':                    # 单小节休止：推进时间，不产生音符
                i += 1
                raw, i = self._parse_dur(s, i)
                t += actual_dur(raw)
                continue

            if c in 'ZX':                    # 多小节休止：直接跳过计数
                i += 1
                while i < n and s[i].isdigit(): i += 1
                continue

            i += 1

        return result

    # ---------- 和弦变换检测 ----------

    def _detect_change_by_roots(self, voice_segs: List[str], key_acc: Dict[int, int],
                                sig_count: int, r1_pc: int, r2_pc: int
                                ) -> Optional[Tuple[str, str, Fraction]]:
        """
        将所有声部的带时间戳音符合并到同一时间轴（各声部均从 t=0 起），
        按时间排序后扫描：找到 r2_pc 累积权重首次超过 r1_pc 的时刻为分割点；
        若无明确超越点则以总时长中点作为回退。
        分割后分别判断前/后段最优和弦，若不同则返回 (和弦1, 和弦2)。
        """
        timeline: List[Tuple[Fraction, int, Fraction]] = []
        for seg in voice_segs:
            timeline.extend(self._extract_timed_notes(seg, key_acc))

        if not timeline:
            return None

        timeline.sort(key=lambda x: x[0])
        total_end = max(t + dur for t, _, dur in timeline)

        def _segment_name(pcw: Dict[int, float], preferred_root: int,
                          prefer_complete_seventh: bool = False) -> Optional[str]:
            if not pcw:
                return None
            mode_bias = self._mode_bias_for_weights(pcw, sig_count)
            use_sharps = sum(key_acc.values()) >= 0
            top = self._top_chords(pcw, top_n=1, use_sharps=use_sharps)
            strongest_root = max(pcw.items(), key=lambda x: x[1])[0]
            name = (
                self._best_chord_for_root(
                    pcw, preferred_root, key_acc, mode_bias, sig_count, use_sharps
                )
                or self._best_chord_for_root(
                    pcw, strongest_root, key_acc, mode_bias, sig_count, use_sharps
                )
                or (top[0][0] if top else None)
            )
            if (
                prefer_complete_seventh
                and name
                and name == self._pc_name(preferred_root, use_sharps) + 'm'
                and pcw.get((preferred_root + 7) % 12, 0.0) > 0
                and pcw.get((preferred_root + 10) % 12, 0.0) >= pcw.get(preferred_root, 0.0) * 0.12
            ):
                return name + '7'
            return name

        def _bass_root(notes: List[Tuple[Fraction, int, Fraction]]) -> Optional[int]:
            bass = self._bass_timed_pc_weights(notes)
            if not bass:
                return None
            return max(bass.items(), key=lambda x: x[1])[0]

        def _starts_at(notes: List[Tuple[Fraction, int, Fraction]],
                       pc: Optional[int], t: Fraction) -> bool:
            return pc is not None and any(nt == t and midi % 12 == pc for nt, midi, _ in notes)

        # 优先尝试半小节切分。只有当前后半小节的低音根音分别吻合传入的
        # 两个候选根音时才启用，避免破坏已经正确的非半小节切分。
        midpoint = total_end / 2
        first_half = [(t, midi, dur) for t, midi, dur in timeline if t < midpoint]
        second_half = [(t, midi, dur) for t, midi, dur in timeline if t >= midpoint]
        first_bass = _bass_root(first_half)
        second_bass = _bass_root(second_half)
        whole_pcw = self._timed_pc_weights(timeline)
        use_sharps = sum(key_acc.values()) >= 0
        whole_top = self._top_chords(whole_pcw, top_n=1, use_sharps=use_sharps)
        whole_root = self._name_root_pc(whole_top[0][0]) if whole_top else None
        if (
            first_half and second_half
            and first_bass in (r1_pc, r2_pc)
            and second_bass in (r1_pc, r2_pc)
            and first_bass != second_bass
            and whole_root != first_bass
            and _starts_at(first_half, first_bass, Fraction(0))
            and _starts_at(second_half, second_bass, midpoint)
        ):
            pcw1 = self._timed_pc_weights(first_half)
            pcw2 = self._timed_pc_weights(second_half)
            name1 = _segment_name(pcw1, first_bass)
            name2 = _segment_name(pcw2, second_bass, prefer_complete_seventh=True)
            if name1 and name2 and self._name_root_pc(name1) != self._name_root_pc(name2):
                return name1, name2, midpoint

        # 找根音切换点
        cum_r1 = 0.0
        cum_r2 = 0.0
        split_t: Optional[Fraction] = None
        for t, midi, dur in timeline:
            pc = midi % 12
            w = self._pitch_weight(midi) * float(dur)
            if pc == r1_pc:
                cum_r1 += w
            elif pc == r2_pc:
                cum_r2 += w
            if split_t is None and cum_r2 > cum_r1 and cum_r1 > 0:
                split_t = t   # r2 累积权重刚超过 r1 的那一刻

        if split_t is None:
            # 回退：以总时长中点分割
            split_t = total_end / 2

        first_notes = [(t, midi, dur) for t, midi, dur in timeline if t < split_t]
        second_notes = [(t, midi, dur) for t, midi, dur in timeline if t >= split_t]

        if not first_notes or not second_notes:
            return None

        pcw1 = self._timed_pc_weights(first_notes)
        pcw2 = self._timed_pc_weights(second_notes)
        name1 = _segment_name(pcw1, r1_pc)
        name2 = _segment_name(pcw2, r2_pc)
        if name1 and name2 and name1 != name2:
            return name1, name2, split_t
        return None

    def _detect_inversion_motion(self, voice_segs: List[str], key_acc: Dict[int, int],
                                 chord_name: str, bar_dur: Fraction
                                 ) -> Optional[Tuple[Fraction, int, int]]:
        chord_pcs = self._chord_pcs_for_name(chord_name)
        root = self._name_root_pc(chord_name)
        if root is None or not chord_pcs:
            return None

        timeline: List[Tuple[Fraction, int, Fraction]] = []
        for seg in voice_segs:
            timeline.extend(self._extract_timed_notes(seg, key_acc))
        if not timeline:
            return None

        midpoint = bar_dur / 2
        first = [(t, midi, dur) for t, midi, dur in timeline if t < midpoint]
        second = [(t, midi, dur) for t, midi, dur in timeline if t >= midpoint]
        if not first or not second:
            return None

        def bass_root(notes: List[Tuple[Fraction, int, Fraction]]) -> Optional[int]:
            bass = self._bass_timed_pc_weights(notes)
            return max(bass.items(), key=lambda x: x[1])[0] if bass else None

        first_bass = bass_root(first)
        second_bass = bass_root(second)
        if (
            first_bass in chord_pcs
            and second_bass == root
            and first_bass != root
            and any(t == 0 and midi % 12 == first_bass for t, midi, _ in first)
            and any(t == midpoint and midi % 12 == second_bass for t, midi, _ in second)
        ):
            return midpoint, first_bass, second_bass
        return None

    def _has_bass_pc_at(self, voice_segs: List[str], key_acc: Dict[int, int],
                        pc: int, t: Fraction) -> bool:
        for seg in voice_segs:
            for nt, midi, _ in self._extract_timed_notes(seg, key_acc):
                if nt == t and midi < self._BASS_CUTOFF and midi % 12 == pc:
                    return True
        return False

    # ---------- 主流程 ----------

    def process(self, lines: List[str], auto_fix: bool) -> Tuple[List[Issue], List[str]]:
        issues: List[Issue] = []

        k_idx = next((i for i, l in enumerate(lines)
                      if l.lstrip().startswith('K:')), None)
        if k_idx is None:
            return issues, list(lines)

        key_acc = self._parse_key_acc(lines[:k_idx + 1])
        sig_count = self._parse_key_signature_count(lines[:k_idx + 1])

        # 解析 M: 和 L: 计算每小节时值（单位：L 值）
        meter = Fraction(4, 4)
        unit = Fraction(1, 8)
        for ln in lines:
            s = ln.lstrip()
            mm = re.match(r'M:\s*(\d+)/(\d+)', s)
            if mm:
                meter = Fraction(int(mm.group(1)), int(mm.group(2)))
            lm = re.match(r'L:\s*(\d+)/(\d+)', s)
            if lm:
                unit = Fraction(int(lm.group(1)), int(lm.group(2)))
        bar_dur = meter / unit
        use_sharps = sum(key_acc.values()) >= 0

        # 按声部收集音乐行
        voice_music: Dict[str, List[str]] = {}
        current_vid = '__global__'
        for i in range(k_idx + 1, len(lines)):
            s = lines[i].lstrip()
            if s.startswith('V:'):
                m = re.match(r'V:\s*(\S+)', s)
                current_vid = m.group(1) if m else str(i)
            elif s and not s.startswith('%') and not (len(s) >= 2 and s[1] == ':' and s[0].isalpha()):
                voice_music.setdefault(current_vid, []).append(lines[i])

        # 每个声部切分为小节列表
        voice_measures: Dict[str, List[str]] = {}
        for vid, mlines in voice_music.items():
            combined = ' '.join(l.rstrip() for l in mlines)
            segs = [seg for seg in self._split_measures(combined)
                    if seg.strip() and not seg.strip().startswith('%')]
            if segs:
                voice_measures[vid] = segs

        if not voice_measures:
            return issues, list(lines)

        max_bars = max(len(v) for v in voice_measures.values())

        global_notes: List[Tuple[Fraction, int, Fraction]] = []
        for segs in voice_measures.values():
            for seg in segs:
                global_notes.extend(self._extract_timed_notes(seg, key_acc))
        inferred_key, inferred_mode, key_confidence = self._infer_relative_key(global_notes, sig_count)
        if inferred_mode in ('major', 'minor'):
            issues.append(Issue(
                line_index=k_idx,
                description=(
                    f"推断调性为 {inferred_key}"
                    f"（关系大小调置信比 {key_confidence:.2f}）"
                ),
                severity="info",
            ))

        # 收集每小节的和弦信息，用于生成柱式和弦声部
        # 格式：'change' -> (c1, split_t, c2) | 'single' -> chord_name
        bar_chord_data: Dict[int, tuple] = {}
        bar_pcw: Dict[int, Dict[int, float]] = {}
        bar_bass_pcw: Dict[int, Dict[int, float]] = {}

        for bar_idx in range(max_bars):
            all_notes: List[Tuple[Fraction, int, Fraction]] = []
            bar_segs: List[str] = []
            for segs in voice_measures.values():
                if bar_idx < len(segs):
                    seg = segs[bar_idx]
                    ns = self._extract_timed_notes(seg, key_acc)
                    all_notes.extend(ns)
                    if ns:
                        bar_segs.append(seg)

            if not all_notes:
                continue

            pcw = self._timed_pc_weights(all_notes)
            bar_pcw[bar_idx] = pcw
            mode_bias = self._mode_bias_for_weights(pcw, sig_count)
            if inferred_mode in ('major', 'minor') and mode_bias == 'neutral':
                mode_bias = inferred_mode
            top = self._top_chords(pcw, top_n=2, use_sharps=use_sharps)
            if not top:
                continue

            bar_num = bar_idx + 1
            best_name, best_score = top[0]
            inversion_name = self._prefer_complete_inversion(pcw, best_name, best_score, use_sharps)
            if inversion_name:
                best_name = inversion_name
                best_score = self._score_chord(
                    pcw,
                    self._name_root_pc(best_name) or 0,
                    next((ivs for suf, ivs in self._CHORD_TYPES if suf == self._name_suffix(best_name)), [0, 4, 7])
                )
            best_name = self._simplify_close_seventh(pcw, best_name, best_score, use_sharps)

            # 按权重排序找出前两根音（优先用低音区权重，旋律高音不干扰根音判断）
            bass_pcw = self._bass_timed_pc_weights(all_notes)
            bar_bass_pcw[bar_idx] = bass_pcw
            root_source = bass_pcw if bass_pcw else pcw
            total_w = sum(root_source.values())
            sorted_roots = sorted(root_source.items(), key=lambda x: -x[1])
            r1_pc, r1_w = sorted_roots[0]

            chord_scores_close = (
                len(top) >= 2 and top[1][1] > 0
                and min(top[1][1], best_score) / max(top[1][1], best_score) >= self._CHANGE_RATIO
            )

            # 只有两个根音都足够大且接近时才考虑和弦变换
            roots_competing = False
            r2_pc = r1_pc
            if len(sorted_roots) >= 2:
                r2_pc, r2_w = sorted_roots[1]
                root_ratio = r2_w / r1_w if r1_w > 0 else 0
                roots_competing = (
                    root_ratio >= self._ROOT_RATIO
                    and r1_w / total_w >= self._ROOT_MIN_FRAC
                    and r2_w / total_w >= self._ROOT_MIN_FRAC
                )

            _ROOT_RE = re.compile(r'^([A-Ga-g][#b]?)')
            def _chord_root(name: str) -> str:
                rm = _ROOT_RE.match(name)
                return rm.group(1) if rm else name

            def _set_single_data(name: str) -> None:
                inv = self._detect_inversion_motion(bar_segs, key_acc, name, bar_dur)
                if inv:
                    split_t, inv1, inv2 = inv
                    bar_chord_data[bar_idx] = ('inversion', name, split_t, inv1, inv2)
                else:
                    bar_chord_data[bar_idx] = ('single', name)

            if roots_competing:
                # 低音区两个根音竞争激烈，尝试拆分检测和弦变换（不依赖整体和弦得分是否接近）
                best_pcs = self._chord_pcs_for_name(best_name)
                whole_chord_is_stable = (
                    best_score >= 0.90
                    and r1_pc in best_pcs
                    and r2_pc in best_pcs
                )
                change = self._detect_change_by_roots(
                    bar_segs, key_acc, sig_count, r1_pc, r2_pc
                )
                if whole_chord_is_stable and change:
                    c1_tmp, c2_tmp, split_tmp = change
                    c1_root_tmp = self._name_root_pc(c1_tmp)
                    c2_root_tmp = self._name_root_pc(c2_tmp)
                    stable_inversion = (
                        c1_root_tmp is not None
                        and c2_root_tmp is not None
                        and c2_root_tmp in best_pcs
                        and self._has_bass_pc_at(bar_segs, key_acc, c2_root_tmp, split_tmp)
                    )
                    if not stable_inversion:
                        change = None
                if change and _chord_root(change[0]) != _chord_root(change[1]):
                    c1, c2, split_t = change
                    bar_chord_data[bar_idx] = ('change', c1, split_t, c2)
                    r2_w_val = root_source.get(r2_pc, 0.0)
                    ratio_str = f"{r2_w_val / r1_w:.0%}"
                    desc = (
                        f"第 {bar_num} 小节：和弦变换 {c1} → {c2}"
                        f"（根音 {self._PC_NAMES[r1_pc]}={r1_w:.1f} / "
                        f"{self._PC_NAMES[r2_pc]}={r2_w_val:.1f}，"
                        f"权重比 {ratio_str}）"
                    )
                else:
                    preferred_name = self._prefer_root_chord(
                        pcw, r1_pc, best_name, best_score, key_acc,
                        mode_bias, sig_count, use_sharps
                    )
                    if preferred_name:
                        best_name = preferred_name
                    _set_single_data(best_name)
                    if chord_scores_close:
                        second_name, second_score = top[1]
                        close_ratio = min(second_score, best_score) / max(second_score, best_score)
                        desc = (
                            f"第 {bar_num} 小节：推断和弦 {best_name}"
                            f"（次选 {second_name}，和弦得分接近 {close_ratio:.0%}）"
                        )
                    else:
                        desc = f"第 {bar_num} 小节：推断和弦 {best_name}（得分 {best_score:.2f}）"
            elif chord_scores_close:
                preferred_name = self._prefer_root_chord(
                    pcw, r1_pc, best_name, best_score, key_acc,
                    mode_bias, sig_count, use_sharps
                )
                if preferred_name:
                    best_name = preferred_name
                _set_single_data(best_name)
                second_name, second_score = top[1]
                close_ratio = min(second_score, best_score) / max(second_score, best_score)
                desc = (
                    f"第 {bar_num} 小节：推断和弦 {best_name}"
                    f"（次选 {second_name}，和弦得分接近 {close_ratio:.0%}）"
                )
            else:
                preferred_name = self._prefer_root_chord(
                    pcw, r1_pc, best_name, best_score, key_acc,
                    mode_bias, sig_count, use_sharps
                )
                if preferred_name:
                    best_name = preferred_name
                _set_single_data(best_name)
                desc = f"第 {bar_num} 小节：推断和弦 {best_name}（得分 {best_score:.2f}）"

            # verbose 模式附加根音权重明细
            if self.verbose:
                sorted_pcw = sorted(pcw.items(), key=lambda x: -x[1])
                detail = '  '.join(
                    f"{self._PC_NAMES[pc]}={w:.2f}" for pc, w in sorted_pcw
                )
                desc += f"\n  权重明细：{detail}"
                if bass_pcw:
                    sorted_bass = sorted(bass_pcw.items(), key=lambda x: -x[1])
                    bass_detail = '  '.join(
                        f"{self._PC_NAMES[pc]}={w:.2f}" for pc, w in sorted_bass
                    )
                    desc += f"\n  低音权重：{bass_detail}"

            issues.append(Issue(line_index=k_idx + 1, description=desc, severity="info"))

        if inferred_mode == 'minor':
            _, minor_tonic = self._SIG_RELATIVE_KEYS.get(sig_count, (0, 9))
            major_tonic, _ = self._SIG_RELATIVE_KEYS.get(sig_count, (0, 9))
            dominant = (minor_tonic + 7) % 12
            leading = (minor_tonic + 11) % 12
            subtonic = (minor_tonic + 10) % 12

            def set_single(idx: int, name: str) -> None:
                bar_chord_data[idx] = ('single', name)

            def chord_complete(pcw_i: Dict[int, float], name: str) -> bool:
                pcs = list(self._chord_pcs_for_name(name))
                return bool(pcs) and self._has_all_pcs(pcw_i, pcs)

            def data_roots(data: tuple) -> List[int]:
                if not data:
                    return []
                if data[0] in ('single', 'single_drop_fifth'):
                    root = self._name_root_pc(data[1])
                    return [] if root is None else [root]
                if data[0] == 'change':
                    roots = []
                    for name in (data[1], data[3]):
                        root = self._name_root_pc(name)
                        if root is not None:
                            roots.append(root)
                    return roots
                if data[0] == 'inversion':
                    root = self._name_root_pc(data[1])
                    return [] if root is None else [root]
                return []

            for idx, pcw_i in bar_pcw.items():
                data = bar_chord_data.get(idx)
                if not data or data[0] != 'single':
                    continue
                current = data[1]
                current_score = self._named_score(pcw_i, current)

                dominant7 = self._pc_name(dominant, use_sharps) + '7'
                if self._has_all_pcs(pcw_i, [dominant, dominant + 4, dominant + 7, dominant + 10]):
                    dom_score = self._named_score(pcw_i, dominant7)
                    root_w = pcw_i.get(dominant, 0.0)
                    third_w = pcw_i.get((dominant + 4) % 12, 0.0)
                    seventh_w = pcw_i.get((dominant + 10) % 12, 0.0)
                    if (
                        third_w >= max(0.35, root_w * 0.03)
                        and seventh_w >= max(0.35, root_w * 0.18)
                        and (
                            self._name_root_pc(current) == dominant
                            or dom_score >= current_score * 0.90
                            or (idx > 0 and dominant in data_roots(bar_chord_data.get(idx - 1, ())))
                        )
                    ):
                        set_single(idx, dominant7)
                        continue

                dominant_major = self._pc_name(dominant, use_sharps)
                if self._has_all_pcs(pcw_i, [dominant, dominant + 4, dominant + 7]):
                    dom_major_score = self._named_score(pcw_i, dominant_major)
                    root_w = pcw_i.get(dominant, 0.0)
                    third_w = pcw_i.get((dominant + 4) % 12, 0.0)
                    if (
                        third_w >= max(0.35, root_w * 0.03)
                        and (
                            self._name_root_pc(current) == dominant
                            or dom_major_score >= current_score * 0.90
                        )
                    ):
                        set_single(idx, dominant_major)
                        continue

                leading_dim = self._pc_name(leading, use_sharps) + 'dim'
                if self._has_all_pcs(pcw_i, [leading, leading + 3, leading + 6]):
                    dim_score = self._named_score(pcw_i, leading_dim)
                    leading_w = pcw_i.get(leading, 0.0)
                    current_root = self._name_root_pc(current)
                    current_root_w = pcw_i.get(current_root, 0.0) if current_root is not None else 0.0
                    dim_pcs = {(leading + iv) % 12 for iv in (0, 3, 6)}
                    if (
                        current_root in dim_pcs
                        and leading_w >= max(0.25, current_root_w * 0.16)
                        and dim_score >= current_score * 0.62
                    ):
                        set_single(idx, leading_dim)
                        continue

                subtonic_name = self._pc_name(subtonic, use_sharps)
                bass = bar_bass_pcw.get(idx, {})
                if bass and max(bass.items(), key=lambda x: x[1])[0] == subtonic:
                    sub_score = self._named_score(pcw_i, subtonic_name)
                    if sub_score >= current_score * 0.55:
                        set_single(idx, subtonic_name)

            last_idx = max(bar_pcw) if bar_pcw else None
            if last_idx is not None:
                pcw_last = bar_pcw[last_idx]
                rel_major = self._pc_name(major_tonic, use_sharps)
                last_data = bar_chord_data.get(last_idx, ())
                current_roots = data_roots(bar_chord_data.get(last_idx, ()))
                if last_data and last_data[0] == 'single' and current_roots == [major_tonic]:
                    bar_chord_data[last_idx] = ('single_drop_fifth', rel_major)
                else:
                    if (
                        major_tonic not in current_roots
                        and self._has_all_pcs(pcw_last, [major_tonic, major_tonic + 4, major_tonic + 7])
                    ):
                        bar_chord_data[last_idx] = ('single_drop_fifth', rel_major)

        # 生成柱式和弦声部
        result_lines = list(lines)
        if auto_fix and inferred_mode in ('major', 'minor') and inferred_key:
            key_line = result_lines[k_idx]
            km = re.match(r'(\s*K:\s*)(\S+)(.*)$', key_line)
            if km and km.group(2) != inferred_key:
                result_lines[k_idx] = self._replace_key_field_key(key_line, inferred_key)
            result_lines = self._ensure_voice_key_fields(result_lines, k_idx, inferred_key)

        if self.chord_voice and bar_chord_data:
            while result_lines and not result_lines[-1].strip():
                result_lines.pop()

            existing_vids = set()
            for ln in lines:
                vm = re.match(r'V:\s*(\d+)', ln.lstrip())
                if vm:
                    existing_vids.add(int(vm.group(1)))
            new_vid = max(existing_vids, default=0) + 1

            measures_abc: List[str] = []
            for bi in range(max_bars):
                if bi in bar_chord_data:
                    data = bar_chord_data[bi]
                    if data[0] == 'change':
                        _, c1, split_t, c2 = data
                        dur1 = split_t
                        dur2 = bar_dur - dur1
                        if dur1 <= 0 or dur2 <= 0:
                            dur1 = dur2 = bar_dur / 2

                        c1_root = self._name_root_pc(c1)
                        c2_root = self._name_root_pc(c2)
                        c1_suffix = self._name_suffix(c1)
                        c1_ivs = next(
                            (ivs for suf, ivs in self._CHORD_TYPES if suf == c1_suffix),
                            [0, 4, 7],
                        )
                        c1_pcs = (
                            {(c1_root + iv) % 12 for iv in c1_ivs}
                            if c1_root is not None else set()
                        )
                        c2_is_c1_inversion = (
                            c1_root is not None
                            and c2_root is not None
                            and c2_root != c1_root
                            and c2_root in c1_pcs
                        )
                        c2_suffix = self._name_suffix(c2)
                        c2_ivs = next(
                            (ivs for suf, ivs in self._CHORD_TYPES if suf == c2_suffix),
                            [0, 4, 7],
                        )
                        c2_pcs = (
                            {(c2_root + iv) % 12 for iv in c2_ivs}
                            if c2_root is not None else set()
                        )
                        c1_is_c2_inversion = (
                            c1_root is not None
                            and c2_root is not None
                            and c1_root != c2_root
                            and c1_root in c2_pcs
                            and c1_suffix in ('aug', 'dim', 'dim7')
                        )

                        if c1_is_c2_inversion:
                            s = self._chord_name_to_abc(
                                c2, dur1, key_acc, use_sharps,
                                inversion_pc=c1_root, drop_root=True
                            )
                            s += self._chord_name_to_abc(c2, dur2, key_acc, use_sharps)
                        else:
                            s = self._chord_name_to_abc(c1, dur1, key_acc, use_sharps)
                        if (not c1_is_c2_inversion) and c2_is_c1_inversion:
                            s += self._chord_name_to_abc(
                                c1, dur2, key_acc, use_sharps, inversion_pc=c2_root
                            )
                        elif not c1_is_c2_inversion:
                            s += self._chord_name_to_abc(c2, dur2, key_acc, use_sharps)
                    elif data[0] == 'inversion':
                        _, chord_name, split_t, inv1, inv2 = data
                        dur1 = split_t
                        dur2 = bar_dur - dur1
                        s = (
                            self._chord_name_to_abc(
                                chord_name, dur1, key_acc, use_sharps, inversion_pc=inv1
                            )
                            + self._chord_name_to_abc(
                                chord_name, dur2, key_acc, use_sharps, inversion_pc=inv2
                            )
                        )
                    elif data[0] == 'single_drop_fifth':
                        _, chord_name = data
                        s = self._chord_name_to_abc(
                            chord_name, bar_dur, key_acc, use_sharps, drop_fifth=True
                        )
                    else:
                        _, chord_name = data
                        s = self._chord_name_to_abc(chord_name, bar_dur, key_acc, use_sharps)
                else:
                    s = 'z' + self._dur_to_str(bar_dur)
                measures_abc.append(s)

            # 从第一个有实质音符的声部检测每行小节数模式
            _bar_re = re.compile(r':\|:|\|\]|\|\||\|:|\|')
            bars_per_line: List[int] = []
            for vid, mlines in voice_music.items():
                if not any(re.search(r'[A-Ga-g]', l) for l in mlines):
                    continue  # 跳过全休止声部
                for mline in mlines:
                    stripped = mline.strip()
                    if not stripped or stripped.startswith('%'):
                        continue
                    n = len(_bar_re.findall(stripped))
                    if n > 0:
                        bars_per_line.append(n)
                if bars_per_line:
                    break
            if not bars_per_line:
                bars_per_line = [4]  # 默认每行 4 小节

            music_lines: List[str] = []
            buf: List[str] = []
            pat_idx = 0
            for bi, seg in enumerate(measures_abc):
                buf.append(seg)
                limit = bars_per_line[pat_idx] if pat_idx < len(bars_per_line) else bars_per_line[-1]
                is_last_bar = (bi == len(measures_abc) - 1)
                if len(buf) >= limit or is_last_bar:
                    line = ' | '.join(buf) + (' |]' if is_last_bar else ' |')
                    music_lines.append(line)
                    buf = []
                    pat_idx += 1

            # 检测当前行列表是否带行尾 \n（readlines 模式 vs split("\n") 模式）
            has_nl = any(l.endswith('\n') for l in lines if l)
            nl = '\n' if has_nl else ''

            # 确保和前一个声部之间有换行分隔（不产生额外空行）
            if has_nl and result_lines and not result_lines[-1].endswith('\n'):
                result_lines[-1] = result_lines[-1] + '\n'

            unit_str = f'{unit.numerator}/{unit.denominator}'
            new_voice_lines = [
                f'V:{new_vid} treble nm="Harmony"{nl}',
                f'%%MIDI program 0{nl}',
                f'L:{unit_str}{nl}',
            ]
            if auto_fix and inferred_key:
                new_voice_lines.insert(1, f'K:{inferred_key}{nl}')
            new_voice_lines.extend(
                ln.rstrip('\n').rstrip() + nl for ln in music_lines
            )
            result_lines.extend(new_voice_lines)

        return issues, result_lines
