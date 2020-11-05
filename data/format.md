# Data format

xxx.pkl: original data with ids and single intent
xxx_multi.pkl: original data with ids and multi intents
xxx_raw.pkl: original data with plain texts
xxx_pretrain.pkl: pretrain data

## ATIS (Single Intent)

>
    1. raw_data: 
    [(sentence1, intent_id1, slot1),
     (sentence2, intent_id2, slot2), ...]
    
    sentence: tokenized or untokenized tokens
    intent_id: intent id
    slot: [{'start': xxx, 'end': xxx, 'value': xxx, 'entity_name': xxx}, ...]

    2. intent2id:
    {intent: id}

## TOP (Multiple Intent)

>
    1. raw_data: 
    [(sentence1, intent_ids1),
     (sentence2, intent_ids2), ...]
    
    sentence: tokenized or untokenized tokens
    intent_ids: a list of intent ids

    2. intent2id:
    {intent: id}

    3. intent2id_multi_with_tokens:
    {intent: (id, tokenized_id)}

## Microsoft e2e-dialogue (utterance-based Dialogue)

>
    1. train_data:
    Every dialogue:
    [(sent1, [intent1, intent2], [slot1, slot2]), 
     (sent2, [intent1], [slot1]),...]
    
    2. intent2id:
    {intent: id}

    3. intent2id_multi:
    {intent: id}

    4. intent2id_multi_with_tokens:
    {intent: (id, tokenized_id)}

## SGD dataset (utterance-based+turn-based Dialogue)

>
    1. train_data:
    Every dialogue:
    [(sent1, [intent1, intent2], [slot1, slot2]), 
     (sent2, [intent1], [slot1]),...]
    
    2. intent2id:
    {intent: id}

    3. intent2id_multi:
    {intent: id}

    4. intent2id_multi_with_tokens:
    {intent: (id, tokenized_id)}

    5. turn_data_all = {'turns': all_data_turn,
                        'aintent2id': aintent2id,
                        'request2id': request2id,
                        'slot2id': slot2id,
                        'value2id': value2id}
        all_data_turn: a list of dialogues (turn-level)
        for each dialogue:
            [(turn1, intents1, requested_slots1, slots1, values1, (sent1's data, sent2's data)),...
             (turn2, intents2, requested_slots2, slots2, values2, (sent1's data, sent2's data)),...]

## MultiWOZ2.1 (turn-based Dialogue)

>
    1. turn_data_all = {'turns': all_data_turn,
                        'aintent2id': aintent2id,
                        'request2id': request2id,
                        'slot2id': slot2id,
                        'value2id': value2id}
    
    all_data_turn: a list of dialogues (turn-level)
    [(topic1, dialogue1), (topic2, dialogue2), ...]

    for each dialogue:
        [(turn1, [(intents1, slots1, values1),
                  (intents2, slots2, values2)...]
         (turn2, [(intents1, slots1, values1),
                  (intents2, slots2, values2)...]
         ...]

## Pretrain dataset

>   
    1. all_data:
    [(encoded['input_ids'], encoded['token_type_ids'], encoded['attention_mask'], label),
     ... ]





