import pandas as pd
import numpy as np
import re

from tqdm import tqdm


# I/O files
descs_path = '/usr0/home/mamille2/tumblr/data/bootstrapped_list_descriptions_recent100.pkl'
outpath = '/usr0/home/mamille2/tumblr/data/bootstrapped_list_descriptions_recent100.pkl'

states_path = '/usr0/home/mamille2/tumblr/data/states.csv'
nationalities_path = '/usr0/home/mamille2/tumblr/data/nationalities.txt'
ethnicities_path = '/usr0/home/mamille2/tumblr/data/ethnicities.txt'


def has_category(cat, segments, terms_re):
    ans = False
    
    if not isinstance(segments, list):
        return ans
    
    ans = any(re.search(terms_re[cat], s) for s in segments)
            
    if cat in excl_terms:
        for c in excl_terms[cat]:
            if any(c in s for s in segments):
                ans = False
            
    return ans


# # Pattern matching for mentions of identity categories

# Load US states
states = [s.lower() for s in pd.read_csv(states_path)['State'].tolist()]

# Load nationalities
with open(nationalities_path) as f:
    nats = [nat.lower() for nat in f.read().splitlines() if (len(nat) > 3 and not nat in states)]

# Load ethnicities
outlist = states + ['coast']
with open(ethnicities_path) as f:
    eths = [e.split()[0].lower() for e in f.read().splitlines() if (len(e.split()[0]) > 4 and not e.split()[0].lower() in outlist)]

# Regex patterns
terms = {
        'age': [r'(?:[^-+\w]|^)([1-6]{1}[0-9]{1})[^-+0-9]|^([1-6]{1}[0-9]{1})$',
               r'twelve',
               r'thirteen',
               r'fourteen',
               r'fifteen',
               r'sixteen',
               r'seventeen',
               r'eighteen',
               r'nineteen',
               r'twenty',
               r'thirty',
               r'forty',
               r'fifty',
               r'sixty'],
#         'location': [],
        'gender': [r'male\b', r'female', 
                    r'trans', r'ftm', r'mtf', r'cis',
                    r'girl\b', r'boy\b', r'\bman\b', r'guy\b', r'woman', r'gu+rl', r'gii+rl',
                    r'non-binary', r'nonbinary', r'nb', r'agender', r'neutrois',
                    r'\bmom\b', r'\bdad\b', r'wife', r'husband', r'\bbrother\b', r'\bson\b', r'\bsister\b',
                    r'bigender', r'lgbt'],
        'sexual orientation': 
                     [r'gay', r'straight', r'lesbian', r'\bhomo',
                       r'bisexual', r'\bbi\b', r'pansexual', r'\bpan\b',
                       r'lgbt', r'queer',
                       r'\bace\b', r'\basexual', r'aro-ace', r'aro/ace',
                     ],
         'pronouns': [
             r'(?:\W|\b)she(?:\W|\b)', r'(?:\W|\b)her(?:\W|\b)',
             r'(?:\W|\b)he(?:\W|\b)', r'(?:\W|\b)him(?:\W|\b)',
             r'(?:\W|\b)they(?:\W|\b)', r'(?:\W|\b)them(?:\W|\b)',
             r'pronouns'
                ],
        'personality type': [
            r'(?:i|e|a)(?:s|n)(?:t|f)(?:j|p)',
            r'introvert',
            r'extrovert', 
            r'ambivert',
            r'\b[0-9]w[0-9]\b',
            ],
        'ethnicity/nationality': [r'\b{}\b'.format(el) for el in eths + nats] + 
                [r'latino', r'latina', r'cubana', r'cubano', r'chilena', r'chileno', r'mexicano', r'mexicana',
                r'palestinian'],
        'relationship status': [
            r'taken', r'married', r'single', r'engaged', r'husband', r'spouse', r'wife', r'newlywed',
            r'in a rl', r'in rl', r'in a relationship',
        ]
}
terms['sexuality/gender'] = terms['gender'] + terms['sexual orientation'] + terms['pronouns']

excl_terms = {
    'age': ['nsfw 18', '18 nsfw', '18 only', 'only 18', '18+'],
}

# Combine terms in regex
terms_re = {}
for cat in terms:
    terms_re[cat] = r'|'.join(terms[cat])

# ## Apply to corpus of descriptions

# Load blog descriptions
print("Loading blog descriptions...")
descs = pd.read_pickle(descs_path)
print(descs.columns)
len(descs)

# Annotate for identity categories
print("Annotating identity categories...")
for cat in terms:
    print(cat)
    #descs[cat] = descs['segments_25_nopunct'].map(lambda x: has_category(cat, x, terms_re))
    descs[cat] = list(map(lambda x: has_category(cat, x, terms_re), tqdm(descs['segments_25_nopunct'].tolist())))

# Save annotated data
pd.to_pickle(outpath)
print("Saved annotated data to {}".format(outpath))
