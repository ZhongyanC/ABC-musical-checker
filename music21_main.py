"""
music21-based ABC notation checker / auto-fixer.

Mirrors the four-module pipeline in main.py but delegates parsing and
music-theory analysis to the music21 library.

Install dependencies:
    pip install music21

For image rendering, one of the following must be installed:
    - MuseScore 3/4  (recommended)  https://musescore.org/
    - LilyPond                       https://lilypond.org/

Key differences from main.py:
  - Clef selection uses music21.clef.bestClef() rather than a manual
    MIDI-range comparison.
  - Measure-duration arithmetic is driven by music21's Duration objects
    (quarterLength) rather than Fraction arithmetic over ABC syntax.
  - Header checking stays text-based because music21 does not expose
    raw ABC header tokens (X:, L:, etc.) after parsing.
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

try:
    from music21 import clef as m21_clef, converter, environment, meter, note, stream
except ImportError:
    raise SystemExit("music21 未安装，请先执行: pip install music21")


# ---------------------------------------------------------------------------
# Shared data structure
# ---------------------------------------------------------------------------

@dataclass
class Issue:
    line_index: int   # part index (0-based) or -1 for global issues
    description: str
    severity: str     # "error" | "warning" | "info"


# ---------------------------------------------------------------------------
# Module 1 – Header Checker  (text-based)
# ---------------------------------------------------------------------------

class HeaderChecker:
    """
    Checks that required ABC header fields are present in the raw text.
    music21 does not surface all raw header tokens after parsing, so
    this module operates directly on the ABC string.
    """

    def __init__(self, required_headers_with_defaults: Dict[str, str]):
        self.required = required_headers_with_defaults

    def check(self, abc_text: str) -> List[Issue]:
        found: set = set()
        for line in abc_text.splitlines():
            if len(line) >= 2 and line[1] == ':':
                found.add(line[0])
        return [
            Issue(-1, f"缺失必要的头部字段 {field}:", "error")
            for field in self.required
            if field not in found
        ]

    def fix(self, abc_text: str) -> str:
        found: set = set()
        lines = abc_text.splitlines()
        for line in lines:
            if len(line) >= 2 and line[1] == ':':
                found.add(line[0])
        prepend = [
            f"{field}:{default}"
            for field, default in self.required.items()
            if field not in found
        ]
        return '\n'.join(prepend + lines) if prepend else abc_text


# ---------------------------------------------------------------------------
# Module 2 – Clef Auto-Selector  (music21 bestClef)
# ---------------------------------------------------------------------------

class ClefAutoSelector:
    """
    Uses music21.clef.bestClef() to determine the optimal clef for each
    part and, optionally, replaces the clef at the start of the part.
    """

    def check(self, score: stream.Score) -> List[Issue]:
        issues: List[Issue] = []
        for idx, part in enumerate(score.parts):
            notes_in_part = list(part.recurse().getElementsByClass(note.Note))
            if not notes_in_part:
                continue

            existing = list(part.recurse().getElementsByClass(m21_clef.Clef))
            current_name = type(existing[0]).__name__ if existing else "TrebleClef"

            suggested = m21_clef.bestClef(part.flatten())
            if suggested is None:
                continue
            suggested_name = type(suggested).__name__

            if suggested_name != current_name:
                midi_vals = [n.pitch.midi for n in notes_in_part]
                avg = sum(midi_vals) / len(midi_vals)
                issues.append(Issue(
                    idx,
                    (f"声部 {idx + 1} 平均音高 MIDI={avg:.1f}，"
                     f"当前谱号 {current_name}，建议改用 {suggested_name}"),
                    "info",
                ))
        return issues

    def fix(self, score: stream.Score) -> stream.Score:
        for part in score.parts:
            notes_in_part = list(part.recurse().getElementsByClass(note.Note))
            if not notes_in_part:
                continue
            suggested = m21_clef.bestClef(part.flatten())
            if suggested is None:
                continue
            measures = list(part.getElementsByClass(stream.Measure))
            if not measures:
                continue
            first = measures[0]
            for old in list(first.getElementsByClass(m21_clef.Clef)):
                first.remove(old)
            first.insert(0.0, suggested)
        return score


# ---------------------------------------------------------------------------
# Module 3 – Voice / Part Bar-Count Checker
# ---------------------------------------------------------------------------

class VoiceBarCountChecker:
    """
    Checks that every part (voice) contains the same number of non-empty
    measures.  Auto-fix pads shorter parts with full-bar rests.
    """

    @staticmethod
    def _count(part: stream.Part) -> int:
        return sum(
            1 for m in part.getElementsByClass(stream.Measure)
            if list(m.notesAndRests)
        )

    def check(self, score: stream.Score) -> List[Issue]:
        counts = [self._count(p) for p in score.parts]
        if len(set(counts)) <= 1:
            return []
        max_c = max(counts)
        return [
            Issue(
                idx,
                (f"声部 {idx + 1} 共 {cnt} 个小节，"
                 f"少于最长声部的 {max_c} 个（缺 {max_c - cnt} 小节）"),
                "warning",
            )
            for idx, cnt in enumerate(counts)
            if cnt < max_c
        ]

    def fix(self, score: stream.Score) -> stream.Score:
        counts = [self._count(p) for p in score.parts]
        if len(set(counts)) <= 1:
            return score
        max_c = max(counts)
        for part, cnt in zip(score.parts, counts):
            if cnt >= max_c:
                continue
            ts_list = list(part.recurse().getElementsByClass(meter.TimeSignature))
            bar_ql = ts_list[0].barDuration.quarterLength if ts_list else 4.0
            for _ in range(max_c - cnt):
                m = stream.Measure()
                m.append(note.Rest(quarterLength=bar_ql))
                part.append(m)
        return score


# ---------------------------------------------------------------------------
# Module 4 – Measure Duration Checker
# ---------------------------------------------------------------------------

class MeasureDurationChecker:
    """
    Verifies that each measure's total note/rest duration matches the
    prevailing time signature, with special handling for pickup (弱起) measures.

    Pickup detection
    ----------------
    A *global pickup* is declared when ALL parts satisfy both conditions:
      1. The first measure's duration does not equal the full bar duration.
      2. Every part's first measure has the same duration as every other part's.

    Global pickup → first measures are left untouched during both check and fix.

    Non-global pickup → first measures are treated as ordinary measures but the
    repair target for each part is max(first-measure durations across all parts)
    rather than the full bar duration.  Shorter first measures are padded to that
    target; longer ones are trimmed.

    Measures 2+ are always fixed against the full bar duration.
    """

    _EPS = 1e-6

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _measure_ql(self, measure: stream.Measure) -> float:
        return float(sum(el.duration.quarterLength for el in measure.notesAndRests))

    def _part_ts(self, part: stream.Part) -> Optional[meter.TimeSignature]:
        """
        Return the effective TimeSignature for a part.
        music21 places the TS in the first *full* measure (M1), not in a
        pickup measure (M0), so getContextByClass fails on M0.
        We search all measures in forward order instead.
        """
        for m in part.getElementsByClass(stream.Measure):
            ts = m.getContextByClass(meter.TimeSignature)
            if ts is not None:
                return ts
        return None

    def _first_measures(self, score: stream.Score) -> List[Optional[stream.Measure]]:
        """Return the first non-empty measure per part (None if the part is empty)."""
        result = []
        for part in score.parts:
            first = next(
                (m for m in part.getElementsByClass(stream.Measure)
                 if list(m.notesAndRests)),
                None,
            )
            result.append(first)
        return result

    def _detect_pickup(self, score: stream.Score) -> Tuple[bool, float]:
        """
        Returns (is_global_pickup, first_measure_target_ql).

        is_global_pickup   – True when all parts share the same sub-bar first duration.
        first_measure_target_ql – duration to align non-pickup first measures to
                                  (= max of all first-measure durations).
        """
        firsts = self._first_measures(score)
        durations = []
        for m, part in zip(firsts, score.parts):
            if m is None:
                continue
            ts = self._part_ts(part)
            if ts is None:
                continue
            ql = self._measure_ql(m)
            expected = ts.barDuration.quarterLength
            durations.append((ql, expected))

        if not durations:
            return False, 0.0

        first_qls = [ql for ql, _ in durations]

        all_same_duration = len(set(round(q, 9) for q in first_qls)) == 1
        all_short = all(
            abs(ql - exp) > self._EPS and ql < exp
            for ql, exp in durations
        )
        is_global_pickup = all_same_duration and all_short

        target_ql = max(first_qls)
        return is_global_pickup, target_ql

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self, score: stream.Score) -> List[Issue]:
        issues: List[Issue] = []
        is_global_pickup, first_target_ql = self._detect_pickup(score)
        first_measures = self._first_measures(score)

        if is_global_pickup:
            issues.append(Issue(
                -1,
                f"所有声部第一小节时值均为 {first_target_ql:.4g} 拍且一致，识别为弱起小节，跳过检查",
                "info",
            ))

        for part_idx, part in enumerate(score.parts):
            first_m = first_measures[part_idx]
            ts = self._part_ts(part)
            if ts is None:
                continue
            bar_ql = ts.barDuration.quarterLength

            for measure in part.getElementsByClass(stream.Measure):
                if not list(measure.notesAndRests):
                    continue

                is_first = (measure is first_m)

                if is_first and is_global_pickup:
                    continue  # 弱起，跳过

                expected = first_target_ql if is_first else bar_ql
                actual = self._measure_ql(measure)

                if abs(actual - expected) > self._EPS:
                    label = "弱起对齐目标" if is_first else "期望"
                    issues.append(Issue(
                        part_idx,
                        (f"声部 {part_idx + 1} 第 {measure.number} 小节"
                         f"时值 {actual:.4g} 拍，{label} {expected:.4g} 拍"),
                        "warning",
                    ))
        return issues

    @staticmethod
    def _has_tuplets(measure: stream.Measure) -> bool:
        """Return True if the measure contains any tuplet notes."""
        return any(
            bool(el.duration.tuplets)
            for el in measure.recurse().notesAndRests
        )

    def fix(self, score: stream.Score) -> stream.Score:
        is_global_pickup, first_target_ql = self._detect_pickup(score)
        first_measures = self._first_measures(score)

        for part_idx, part in enumerate(score.parts):
            first_m = first_measures[part_idx]
            ts = self._part_ts(part)
            if ts is None:
                continue
            bar_ql = ts.barDuration.quarterLength

            for measure in part.getElementsByClass(stream.Measure):
                if not list(measure.notesAndRests):
                    continue

                is_first = (measure is first_m)

                if is_first and is_global_pickup:
                    continue  # 弱起，不修复

                # 含连音符的小节跳过修复（截断会产生无效的分数时值）
                if self._has_tuplets(measure):
                    continue

                expected = first_target_ql if is_first else bar_ql
                actual = self._measure_ql(measure)
                diff = expected - actual

                if abs(diff) <= self._EPS:
                    continue
                if diff > 0:
                    measure.append(note.Rest(quarterLength=diff))
                else:
                    excess = -diff
                    for el in reversed(list(measure.notesAndRests)):
                        if excess <= self._EPS:
                            break
                        el_ql = el.duration.quarterLength
                        if el_ql <= excess + self._EPS:
                            measure.remove(el)
                            excess -= el_ql
                        else:
                            el.duration.quarterLength = el_ql - excess
                            excess = 0.0
        return score


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

class ABCProcessor:
    """
    Two-phase pipeline:
      Phase 1 (text)    – HeaderChecker
      Phase 2 (music21) – ClefAutoSelector → VoiceBarCountChecker
                          → MeasureDurationChecker

    run()            – returns (issues, score)
    export_musicxml() – saves MusicXML to disk
    render_image()   – renders PNG via MuseScore or LilyPond
    show_score()     – render + open with system viewer
    """

    def __init__(self, required_headers: Optional[Dict[str, str]] = None):
        self.header_checker = HeaderChecker(required_headers or {})
        self.clef_selector  = ClefAutoSelector()
        self.bar_counter    = VoiceBarCountChecker()
        self.dur_checker    = MeasureDurationChecker()

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def run(self,
            abc_text: str,
            auto_fix: bool = False,
            ) -> Tuple[List[Issue], stream.Score]:
        """
        Parse *abc_text*, run all checks, optionally apply fixes.

        Returns
        -------
        issues : List[Issue]
        score  : music21.stream.Score  (fixed if auto_fix=True, else original)
        """
        all_issues: List[Issue] = []

        # Phase 1 – header check (text level)
        all_issues.extend(self.header_checker.check(abc_text))
        if auto_fix:
            abc_text = self.header_checker.fix(abc_text)

        # Parse with music21
        try:
            score = converter.parse(abc_text, format='abc')
        except Exception as exc:
            all_issues.append(Issue(-1, f"music21 解析失败: {exc}", "error"))
            return all_issues, stream.Score()

        # Phase 2 – music21-level checks
        all_issues.extend(self.clef_selector.check(score))
        all_issues.extend(self.bar_counter.check(score))
        all_issues.extend(self.dur_checker.check(score))

        if auto_fix:
            score = self.clef_selector.fix(score)
            score = self.bar_counter.fix(score)
            score = self.dur_checker.fix(score)

        return all_issues, score

    # ------------------------------------------------------------------
    # Export helpers
    # ------------------------------------------------------------------

    @staticmethod
    def export_musicxml(score: stream.Score, path: str) -> str:
        """
        Write *score* to a MusicXML file.

        Parameters
        ----------
        score : music21.stream.Score
        path  : destination file path (e.g. "output.xml")

        Returns
        -------
        Absolute path of the written file.
        """
        out = score.write('musicxml', fp=path)
        return os.path.abspath(str(out))

    @staticmethod
    def render_image(score: stream.Score,
                     path: str,
                     open_after: bool = False,
                     ) -> Optional[str]:
        """
        Render *score* to a PNG image.

        Strategy
        --------
        1. Export score to a temp MusicXML file.
        2. Call MuseScore directly via subprocess (most reliable).
        3. If MuseScore is not found, fall back to music21's LilyPond writer.

        Parameters
        ----------
        score      : music21.stream.Score
        path       : destination PNG file path (e.g. "output.png")
        open_after : if True, open the image with the system viewer on success.

        Returns
        -------
        Absolute path of the rendered image, or None if no renderer is found.
        """
        import tempfile

        abs_png = os.path.abspath(path)

        # --- Step 1: export to a temporary MusicXML file ---
        xml_fd, xml_path = tempfile.mkstemp(suffix='.xml')
        os.close(xml_fd)
        try:
            score.write('musicxml', fp=xml_path)

            # --- Step 2: try MuseScore via subprocess ---
            mscore = _find_musescore()
            if mscore:
                result = subprocess.run(
                    [mscore, '-o', abs_png, xml_path],
                    capture_output=True, timeout=60,
                )
                # MuseScore 4 may write "output-1.png" instead of "output.png"
                candidates = [
                    abs_png,
                    abs_png.replace('.png', '-1.png'),
                ]
                found = next((p for p in candidates if os.path.exists(p)), None)
                if found:
                    if open_after:
                        _open_file(found)
                    return found
                # MuseScore ran but produced no file — read its log for the real error
                err_msg = result.stderr.decode(errors='replace').strip()
                if not err_msg:
                    err_msg = _read_musescore_log_error()
                print(f"[MuseScore 渲染失败] {err_msg or '(无详细信息)'}")

            # --- Step 3: fall back to LilyPond via music21 ---
            try:
                out = score.write('lily.png', fp=path)
                lily_out = str(out)
                candidates = [lily_out, path, lily_out.replace('.png', '-1.png')]
                found = next((p for p in candidates if os.path.exists(p)), None)
                if found:
                    if open_after:
                        _open_file(found)
                    return os.path.abspath(found)
            except Exception:
                pass

        finally:
            if os.path.exists(xml_path):
                os.unlink(xml_path)

        return None

    def show_score(self, score: stream.Score, path: str = 'score_output.png') -> None:
        """
        Render *score* to PNG and open it with the system's default viewer.
        Prints the image path, or a diagnostic message on failure.
        """
        result = self.render_image(score, path, open_after=True)
        if result:
            print(f"乐谱图片已保存并打开: {result}")
        elif _find_musescore() or _find_musescore():
            print(
                "渲染失败：乐谱中含有 MuseScore 无法处理的结构\n"
                "（可能原因：非标准连音符精度、不完整小节等）。\n"
                "MusicXML 文件已保存，可用 MuseScore 手动打开查看。"
            )
        else:
            print(
                "未找到可用的渲染器（MuseScore / LilyPond）。\n"
                "请安装 MuseScore: https://musescore.org/\n"
                "安装后重新运行即可自动渲染乐谱图片。"
            )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_musescore() -> Optional[str]:
    """
    Return the path to the MuseScore executable, or None if not found.
    Checks music21's environment setting first, then common install locations.
    """
    # Check music21 environment
    try:
        us = environment.UserSettings()
        configured = us['musescoreDirectPNGPath']
        if configured and os.path.exists(str(configured)):
            return str(configured)
    except Exception:
        pass

    # Common install locations (Windows)
    candidates = [
        r"C:\Program Files\MuseScore 4\bin\MuseScore4.exe",
        r"C:\Program Files\MuseScore 3\bin\MuseScore3.exe",
        r"C:\Program Files (x86)\MuseScore 3\bin\MuseScore3.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p

    # macOS / Linux: try PATH
    for name in ('mscore', 'musescore', 'MuseScore'):
        result = subprocess.run(
            ['where' if sys.platform == 'win32' else 'which', name],
            capture_output=True,
        )
        if result.returncode == 0:
            return result.stdout.decode().strip().splitlines()[0]

    return None


def _read_musescore_log_error() -> str:
    """
    Read the most recent MuseScore log file and return the most informative
    ERROR line.  MuseScore 4 writes errors to its own log rather than stderr.
    Prefers lines that contain a human-readable message (e.g. 'Incomplete
    measure') over the generic 'failed convert' summary.
    """
    log_dir = os.path.join(
        os.environ.get('LOCALAPPDATA', ''),
        'MuseScore', 'MuseScore4', 'logs',
    )
    try:
        logs = sorted(
            (f for f in os.listdir(log_dir) if f.endswith('.log')),
            reverse=True,
        )
        if not logs:
            return ''
        log_path = os.path.join(log_dir, logs[0])
        with open(log_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        errors = [l.strip() for l in lines if '| ERROR |' in l]
        if not errors:
            return ''
        # Prefer the specific notation error over the generic converter error
        specific = next(
            (e for e in errors if 'fileConvert' in e or 'Incomplete' in e or '[2009]' in e),
            None,
        )
        return specific or errors[-1]
    except Exception:
        return ''


def _open_file(path: str) -> None:
    """Open *path* with the OS default application."""
    try:
        if sys.platform == 'win32':
            os.startfile(path)          # type: ignore[attr-defined]
        elif sys.platform == 'darwin':
            subprocess.run(['open', path], check=False)
        else:
            subprocess.run(['xdg-open', path], check=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    sample_abc = """X:1
T:Rhythmic Logic Stress Test
C:Experimental
M:4/4
L:1/8
%%score {V1 | V2}
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

    required_headers = {'X': '1', 'T': 'Untitled', 'M': '4/4', 'L': '1/4', 'K':'B'}
    engine = ABCProcessor(required_headers=required_headers)

    # --- 纯检查 ---
    print("=== 纯检查模式 ===")
    issues, score = engine.run(sample_abc, auto_fix=True)
    if issues:
        for issue in issues:
            print(f"[{issue.severity}] 声部/行 {issue.line_index} -> {issue.description}")
    else:
        print("（无问题）")

    # --- 自动修复 ---
    print("\n=== 自动修复模式 ===")
    issues, fixed_score = engine.run(sample_abc, auto_fix=True)
    if issues:
        for issue in issues:
            print(f"[{issue.severity}] 声部/行 {issue.line_index} -> {issue.description}")

    # --- 导出 MusicXML ---
    xml_path = engine.export_musicxml(fixed_score, 'output.xml')
    print(f"\nMusicXML 已保存: {xml_path}")

    # --- 渲染乐谱图片 ---
    print("\n正在渲染乐谱图片…")
    engine.show_score(fixed_score, path='output.png')
