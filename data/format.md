# Data format

## ATIS (Single Intent)

>
    raw_data: 
    [(sentence1, intent_id1, slot1),
     (sentence2, intent_id2, slot2), ...]
    
    sentence: tokenized or untokenized tokens
    intent_id: intent id
    slot: [{'start': xxx, 'end': xxx, 'value': xxx, 'entity_name': xxx}, ...]

    intent2id:
    {intent: id}

## TOP (Multiple Intent)

>
    raw_data: 
    [(sentence1, intent_ids1),
     (sentence2, intent_ids2), ...]
    
    sentence: tokenized or untokenized tokens
    intent_ids: a list of intent ids

    intent2id:
    {intent: id}
    





