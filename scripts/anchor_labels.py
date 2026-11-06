"""Question target coordinates provided by the teaching environment only."""
from scripts.wording_task import prompt,PREFIXES

def target_positions(rows,wordings):
    positions=[]
    for row,wording in zip(rows,wordings):
        records='记录:'+''.join(f'{i+1}:{obj}到{place};' for i,(obj,place) in enumerate(row['events']))
        before=PREFIXES[wording[0]] if isinstance(wording,(tuple,list)) else wording.split('{target}')[0]
        positions.append(len((records+before+row['target']).encode('utf-8')))
    return positions
