from string import punctuation

def get_data(fn):
    with open(fn, encoding='utf-8', errors='ignore') as f:
        lines =f.readlines()
        lines.pop(0)
        f.close()
    labels = []
    msgs = []
    for line in lines:
        label = 1 if line[:line.index(',')] == 'spam' else 0
        msg = line[line.index(',')+1:]
        cleaned_msg = clean_msg(msg)
        labels.append(label)
        msgs.append(cleaned_msg)
    return labels,msgs


def clean_msg(msg):
    tokens = msg.split()
    table = str.maketrans('','',punctuation)
    # remove all punctuations
    tokens = [w.translate(table) for w in tokens]
    tokens = [wd.lower() for wd in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short words
    tokens = [w for w in tokens if len(w) > 1]
    return ' '.join(tokens)
msg = u'XXXMobileMovieClub: To use your credit, click the WAP link in the next txt message or click here>> http://wap. xxxmobilemovieclub.com?n=QJKGIGHJJGCBL'
print(clean_msg(msg))
