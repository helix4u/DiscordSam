"""Utility constants for logit bias settings."""
from typing import Dict

# Mapping of token IDs representing the em dash and common variants to a
# strong negative bias. This discourages the model from producing these tokens.
LOGIT_BIAS_EM_DASH: Dict[int, int] = {
      2322: -100,  # '\u2014'
      2733: -100,  # ' \u2014'
      8290: -100,  # '\u2014\u2014'
     20962: -100,  # '\u2014\u2014\u2014\u2014'
     35251: -100,  # '\u2014and'
     41648: -100,  # '\u2014\u2014\u2014\u2014\u2014\u2014\u2014\u2014'
     51692: -100,  # '\u2014the'
     54067: -100,  # '.\u2014'
     65363: -100,  # '\u2014a'
     87643: -100,  # '\u2014\n\n'
     94012: -100,  # '\u2014but'
     94828: -100,  # '\u2014\u2014... (long run)'
     96754: -100,  # '.\u201d\u2014'
    108181: -100,  # '\u2014that'
    114635: -100,  # '\u2014it'
    118256: -100,  # '\u2014in'
    121630: -100,  # '\u2014or'
    121655: -100,  # '\u2014to'
    123101: -100,  # '\u2014\n'
    126952: -100,  # '\u2014I'
    127126: -100,  # '\u201d\u2014'
    134820: -100,  # ' \u2014\n'
    137419: -100,  # '\u2014which'
    140135: -100,  # ' \u2014\u2014'
    142654: -100,  # ' \u2014\n\n'
    144129: -100,  # ')\u2014'
    144787: -100,  # '\u2014is'
    147994: -100,  # ',\u2014'
    155638: -100,  # '\u2014as'
    160984: -100,  # '\u2014not'
    169785: -100,  # '\u2014you'
    178328: -100,  # '\u2014from'
    180500: -100,  # '\u2014including'
    183122: -100,  # '\u2014for'
    183862: -100,  # '\u200b\u2014'
    187349: -100,  # '\u2014they'
    188860: -100,  # '\u2014all'
    190702: -100,  # '\u2014with'
    196615: -100,  # '\u2014we'
    197618: -100,  # '\u2014even'
}

# The OpenAI API expects string keys for the logit_bias mapping.
LOGIT_BIAS_EM_DASH_STR: Dict[str, int] = {str(k): v for k, v in LOGIT_BIAS_EM_DASH.items()}

__all__ = ["LOGIT_BIAS_EM_DASH_STR"]
