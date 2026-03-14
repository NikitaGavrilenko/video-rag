"""
Query Preprocessor for Video Fragment Retrieval
================================================

    from query_preprocessor import QueryPreprocessor
    qp = QueryPreprocessor('train_qa.csv')
    clean = qp("climactIc narratvie seqence")
    # → "climactic narrative sequence"

Pipeline:
  Stage 1  Rule-based normalization   EN+RU   < 0.1ms   CPU
  Stage 2  SymSpell correction        EN      < 0.2ms   CPU
  Stage 3  SAGE T5 correction         RU      ~18ms     GPU (optional)
  +        Foreign transliteration    All     < 0.01ms  CPU

Dependencies:
  Required:  pip install symspellpy pandas
  Optional:  pip install transformers torch sentencepiece  (for SAGE)
"""

import re, os
from collections import Counter
from typing import Optional
import pandas as pd


# ── Constants ─────────────────────────────────────────────

_LAT2CYR = {
    'a':'а','e':'е','o':'о','p':'р','c':'с','x':'х','y':'у',
    'A':'А','E':'Е','O':'О','P':'Р','C':'С','X':'Х','B':'В',
    'H':'Н','K':'К','M':'М','T':'Т',
}
_CYR2LAT = {v: k for k, v in _LAT2CYR.items()}

ARABIC_EN = {
    'ع':'ain','غ':'ghain','ق':'qaf','ك':'kaf','م':'meem','ن':'noon',
    'ث':'tha','س':'seen','ح':'ha','ذ':'dhal','ز':'zay','ض':'dad',
    'ص':'sad','ط':'ta','ظ':'dha','خ':'kha','ج':'jeem','ش':'sheen',
    'ب':'ba','ت':'ta','ف':'fa','ل':'lam','ر':'ra','د':'dal',
    'و':'waw','ي':'ya','ه':'ha',
    'أ':'alif','إ':'alif','آ':'alif','ا':'alif',
    'ة':'ta marbuta','ء':'hamza','ئ':'hamza','ؤ':'hamza',
}
ARABIC_RU = {
    'ع':'айн','غ':'гайн','ق':'каф','ك':'кяф','م':'мим','ن':'нун',
    'ث':'са','س':'син','ح':'ха','ذ':'заль','ز':'зай','ض':'дад',
    'ص':'сад','ط':'та','ظ':'за','خ':'ха','ج':'джим','ش':'шин',
    'ب':'ба','ت':'та','ф':'фа','ل':'лям','ر':'ра','د':'даль',
    'و':'вав','ي':'йа','ه':'ха',
    'أ':'алиф','إ':'алиф','آ':'алиф','ا':'алиф',
    'ة':'та марбута','ء':'хамза','ئ':'хамза','ؤ':'хамза',
}
ARABIC_PHRASES = {
    'يلا بينا':'yalla bina','راح أروح':'rah arooh',
    'سوء إنشاء':'su insha','ماء':'maa','ما':'ma','نا':'na',
}
CHINESE_PHRASES = {
    '幹嘛':'gan ma','沒錯':'mei cuo',
    '我好了':'wo hao le','你慢慢來':'ni man man lai',
}

_MANUAL_WHITELIST = {
    'beatbox','beatboxer','beatboxers','beatboxing',
    'tanween','alif','hamza','makhraj','pharyngeal',
    'emirati','abu','dhabi','dubai','ajman','sharjah',
    'oahu','maui','kauai','haleakala','waikiki','honolulu',
    'foz','cappadocia','istanbul',
    'ender','fdm','sla','pla','petg','gcode','cura',
    'mise','roux','julienne','brunoise',
    'cpr','aed','heimlich','abcs',
    'asl','fingerspelling',
    'gudda','shisha','dupatta','biryani','chai',
    'apps','app','sitch','vibrato','legato','staccato',
    'paradiddle','paradiddles','polyrhythm','polyrhythms',
    'pinyin','hanzi','chun','shan',
}


# ── Helpers ───────────────────────────────────────────────

def detect_lang(text: str) -> str:
    """Detect primary script: 'ru' if mostly Cyrillic, else 'en'."""
    cyr = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
    lat = sum(1 for c in text if 'a' <= c.lower() <= 'z')
    return 'ru' if cyr > lat else 'en'

_RE_QUOTES = re.compile(r'"([^"]*)"')
_RE_ZW = re.compile(r'[\u200b\u200c\u200d\u200e\u200f\ufeff\u00a0\u2060]+')

def _post_sage_clean(original: str, corrected: str) -> str:
    """Fix SAGE cosmetic artifacts: restore guillemets, strip zero-width spaces."""
    if '«' in original and '«' not in corrected:
        corrected = _RE_QUOTES.sub(r'«\1»', corrected)
    corrected = _RE_ZW.sub('', corrected)
    return re.sub(r'\s+', ' ', corrected).strip()


# ── Stage 1: Rule-based normalization ─────────────────────

def normalize(query: str) -> str:
    """Fix mid-word caps, Latin/Cyrillic homoglyphs, merged words."""
    q = query.strip()
    if not q:
        return q

    chars = list(q)
    for i in range(1, len(chars)):
        ch, prev = chars[i], chars[i - 1]
        if not ch.isupper() or not prev.isalpha():
            continue
        if (i + 1 < len(chars) and chars[i + 1].isupper()) or prev.isupper():
            continue
        chars[i] = ch.lower()
    q = ''.join(chars)

    tokens = q.split()
    fixed = []
    for tok in tokens:
        core = re.sub(r'[^\w]', '', tok)
        if not core:
            fixed.append(tok); continue
        nc = sum(1 for c in core if '\u0400' <= c <= '\u04FF')
        nl = sum(1 for c in core if 'a' <= c.lower() <= 'z')
        if nc > 0 and nl > 0:
            m = _LAT2CYR if nc > nl else (_CYR2LAT if nl > nc else {})
            tok = ''.join(m.get(c, c) for c in tok)
        fixed.append(tok)
    q = ' '.join(fixed)

    q = re.sub(r'([a-zа-яё])([A-ZА-ЯЁ][a-zа-яё]{2,})', r'\1 \2', q)
    return re.sub(r'\s+', ' ', q).strip()


# ── Foreign script transliteration ────────────────────────

def transliterate(query: str) -> str:
    """Append phonetic transliteration for Arabic/Chinese characters."""
    if not re.search(r'[\u0600-\u06FF\u4e00-\u9fff]', query):
        return query

    lang = detect_lang(query)
    result = query

    for ph in sorted(ARABIC_PHRASES, key=len, reverse=True):
        if ph in result:
            result = result.replace(ph, ARABIC_PHRASES[ph])
    for ph in sorted(CHINESE_PHRASES, key=len, reverse=True):
        if ph in result:
            result = result.replace(ph, CHINESE_PHRASES[ph])

    diacritics = re.compile(r'[\u064B-\u065F\u0670]')
    lmap = ARABIC_RU if lang == 'ru' else ARABIC_EN

    def _repl_ar(m):
        t = diacritics.sub('', m.group(0))
        parts = [lmap.get(ch, ch) for ch in t if ch.strip()]
        return ' '.join(parts) if parts else m.group(0)

    result = re.sub(
        r'[\u0600-\u06FF\uFB50-\uFDFF\uFE70-\uFEFF\u064B-\u065F]+',
        _repl_ar, result,
    )
    result = re.sub(
        r'[\u4e00-\u9fff]+',
        lambda m: CHINESE_PHRASES.get(m.group(0), m.group(0)), result,
    )
    return f"{query}  {result}" if result != query else query


# ── Stage 2: SymSpell (English only) ─────────────────────

class _SymSpellEN:
    def __init__(self, whitelist: set, max_edit: int = 2):
        from symspellpy import SymSpell, Verbosity
        import pkg_resources
        self.V = Verbosity
        self.max_edit = max_edit
        self.whitelist = whitelist
        self.sym = SymSpell(max_dictionary_edit_distance=max_edit)
        self.sym.load_dictionary(
            pkg_resources.resource_filename(
                "symspellpy", "frequency_dictionary_en_82_765.txt"),
            0, 1,
        )

    def correct(self, query: str) -> str:
        tokens = re.findall(r'[\w]+|[^\w]+', query)
        out = []
        for tok in tokens:
            if not re.match(r'^[a-zA-Z]{3,}$', tok):
                out.append(tok); continue
            if tok.lower() in self.whitelist or tok.isupper():
                out.append(tok); continue
            hits = self.sym.lookup(tok.lower(), self.V.CLOSEST,
                                   max_edit_distance=self.max_edit)
            if hits and hits[0].term != tok.lower():
                fix = hits[0].term
                if tok[0].isupper():
                    fix = fix[0].upper() + fix[1:]
                out.append(fix)
            else:
                out.append(tok)
        return ''.join(out)


# ── Stage 3: SAGE T5 (Russian, optional) ─────────────────

class _SageRU:
    def __init__(self, model="ai-forever/sage-fredt5-distilled-95m",
                 device='auto'):
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        self.tok = AutoTokenizer.from_pretrained(model)
        self.mdl = AutoModelForSeq2SeqLM.from_pretrained(model).to(device).eval()

    def correct(self, query: str) -> str:
        import torch
        with torch.inference_mode():
            enc = self.tok(query, return_tensors="pt",
                           max_length=256, truncation=True).to(self.device)
            out = self.mdl.generate(**enc, max_length=256, num_beams=1)
        return self.tok.decode(out[0], skip_special_tokens=True)

    def correct_batch(self, queries: list[str], bs: int = 16) -> list[str]:
        import torch
        results = []
        with torch.inference_mode():
            for i in range(0, len(queries), bs):
                batch = queries[i:i + bs]
                enc = self.tok(batch, return_tensors="pt", padding=True,
                               max_length=256, truncation=True).to(self.device)
                out = self.mdl.generate(**enc, max_length=256, num_beams=1)
                results.extend(self.tok.batch_decode(out, skip_special_tokens=True))
        return results


# ── Pipeline ──────────────────────────────────────────────

class QueryPreprocessor:
    """
    Three-stage query preprocessing pipeline.

    Args:
        train_csv:    Path to train_qa.csv (auto-builds domain whitelist).
                      None -> uses only built-in manual whitelist.
        use_symspell: Enable SymSpell for English typos.
        use_sage:     Enable SAGE T5 for Russian typos.
        sage_device:  'cuda', 'cpu', or 'auto'.
    """

    def __init__(self, train_csv: Optional[str] = None,
                 use_symspell: bool = True, use_sage: bool = False,
                 sage_device: str = 'auto'):

        whitelist = set(_MANUAL_WHITELIST)
        if train_csv and os.path.exists(train_csv):
            whitelist |= self._build_whitelist(train_csv)

        self.symspell = None
        self.sage = None

        if use_symspell:
            try:
                self.symspell = _SymSpellEN(whitelist)
            except ImportError:
                print("symspellpy not found — skipping EN spell correction")

        if use_sage:
            try:
                self.sage = _SageRU(device=sage_device)
            except ImportError:
                print("transformers/torch not found — skipping SAGE")

    @staticmethod
    def _build_whitelist(csv_path: str) -> set:
        df = pd.read_csv(csv_path)
        texts = []
        for col in ['question_en', 'answer_en', 'topic']:
            texts.extend(df[col].dropna().tolist())
        words = Counter(re.findall(r'[a-zA-Z]{2,}', ' '.join(texts).lower()))
        wl = {w for w, c in words.items() if c >= 2}
        for topic in df['topic'].unique():
            for w in topic.split('-'):
                wl.add(w.lower())
        return wl

    def __call__(self, query: str) -> str:
        q = normalize(query)
        if self.symspell:
            q = self.symspell.correct(q)
        if self.sage and detect_lang(q) == 'ru':
            q = _post_sage_clean(query, self.sage.correct(q))
        return transliterate(q)

    def batch(self, queries: list[str]) -> list[str]:
        processed, ru_idx = [], []
        for i, q in enumerate(queries):
            q = normalize(q)
            if self.symspell:
                q = self.symspell.correct(q)
            processed.append(q)
            if self.sage and detect_lang(q) == 'ru':
                ru_idx.append(i)
        if self.sage and ru_idx:
            ru_qs = [processed[i] for i in ru_idx]
            fixed = self.sage.correct_batch(ru_qs)
            for i, f, orig in zip(ru_idx, fixed, [queries[j] for j in ru_idx]):
                processed[i] = _post_sage_clean(orig, f)
        return [transliterate(q) for q in processed]
