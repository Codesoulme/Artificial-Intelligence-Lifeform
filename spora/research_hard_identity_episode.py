"""Synthetic hard-negative construction for identity research only."""


def hard_episode(name, train_names, known, step, rng):
    if name not in train_names or type(known) is not bool or len(set(train_names))<5:
        raise ValueError('Invalid episode inputs')
    query_name=name
    others=rng.sample([n for n in train_names if n!=name],2 if known else 3)
    if known:
        names=others+[name,name+'甲'];rng.shuffle(names)
    elif step%4==1:
        query_name=name+'甲';names=others+[name];rng.shuffle(names)
    else:
        names=others+[name+'甲'];rng.shuffle(names)
    target=names.index(name) if known else 4
    if len(set(names))!=4 or (not known and query_name in names):
        raise ValueError('Invalid contrastive candidate set')
    return query_name,names,target
