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

## MultiWOZ2.1 (turn-based Dialogue)

>
    train_data:
    Every dialogue:
    [(sent1+sent2, [domain1, domain2]), 
     (sent2+sent3, [domain1]),...]
    
    intent2id:
    {domain: id}

## Microsoft e2e-dialogue (utterance-based Dialogue)

>
    train_data:
    Every dialogue:
    [(sent1, [intent1, intent2], [slot1, slot2]), 
     (sent2, [intent1], [slot1]),...]
    
    intent2id:
    {intent: id}





