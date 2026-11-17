"""Offline threshold audit; does not authorize production identity reads."""
import math


def margin(row):
    scores=row.get('scores')
    if (not isinstance(scores,list) or len(scores)<2
            or any(not isinstance(x,(int,float)) or isinstance(x,bool) or not math.isfinite(x) for x in scores)):
        raise ValueError('Expected finite record and null scores')
    return max(scores[:-1])-scores[-1]


def calibrate(negative_rows,max_false_accepts):
    if type(max_false_accepts) is not int or max_false_accepts<0 or not negative_rows:
        raise ValueError('Expected negatives and nonnegative false-accept limit')
    if max_false_accepts>=len(negative_rows):
        raise ValueError('False-accept limit must be below negative count')
    if any(row.get('known') is not False for row in negative_rows):
        raise ValueError('Calibration rows must be unknown examples')
    ordered=sorted((margin(row) for row in negative_rows),reverse=True)
    # Reads require strict >; this rejects ties at the threshold.
    return ordered[max_false_accepts]


def score(rows,threshold):
    if not isinstance(threshold,(int,float)) or not math.isfinite(threshold):raise ValueError('Invalid threshold')
    known=unknown=correct=wrong_known=reject_known=false_accept=0
    for row in rows:
        scores=row.get('scores');m=margin(row)
        best=max(range(len(scores)-1),key=lambda i:scores[i])
        selected=best if m>threshold else len(scores)-1
        if row.get('known') is True:
            target=row.get('target')
            if type(target) is not int or not 0<=target<len(scores)-1:raise ValueError('Invalid known target')
            known+=1;correct+=int(selected==target);reject_known+=int(selected==len(scores)-1)
            wrong_known+=int(selected!=target and selected!=len(scores)-1)
        elif row.get('known') is False:
            unknown+=1;false_accept+=int(selected!=len(scores)-1)
        else:raise ValueError('Missing known label')
    return {'known':known,'correct_known':correct,'rejected_known':reject_known,'wrong_known':wrong_known,
            'unknown':unknown,'false_accept_unknown':false_accept}
