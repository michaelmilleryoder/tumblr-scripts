import fasttext as ft

INPATH = '/usr0/home/mamille2/tumblr/data/halfday_tokenized_text.txt'
OUTPATH = '/usr0/home/mamille2/tumblr/data/halfday_ft'

print("Training model...")

model = ft.skipgram(INPATH, 
                    OUTPATH, 
                    dim=300,
                    thread=35,
                    silent=0)
