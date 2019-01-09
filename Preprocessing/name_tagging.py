import nltk
print nltk.__version__
from nltk.tag import StanfordNERTagger

import os

st = StanfordNERTagger(os.environ.get('STANFORD_MODELS'))

print st._stanford_jar

stanford_dir = st._stanford_jar[0].rpartition('/')[0]

from nltk.internals import find_jars_within_path
stanford_jars = find_jars_within_path(stanford_dir)

print ":".join(stanford_jars)

print st._stanford_jar

print st.tag('Manash is awesome'.split())
