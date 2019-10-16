from bert_serving.client import BertClient
bert = BertClient(ip='localhost',show_server_config=True) # ip为服务器ip地址，本机为localhost或不写
test_vector = bert.encode(["今天天气很好","这是一个示例"])
print(test_vector)