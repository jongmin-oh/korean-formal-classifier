import re
import emoji
from soynlp.normalizer import repeat_normalize

pattern = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-ㅣ가-힣]+')
url_pattern = re.compile(
    r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
jamo_pattern = re.compile('[|ㄱ-ㅎ|ㅏ-ㅣ]+')


def clean(x):
    x = pattern.sub(' ', x)
    x = emoji.replace_emoji(x, replace='')  # emoji 삭제
    x = url_pattern.sub('', x)
    x = jamo_pattern.sub('', x)
    x = x.strip()
    x = repeat_normalize(x, num_repeats=2)
    return x
