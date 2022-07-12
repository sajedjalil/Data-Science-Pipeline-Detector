def get_roberta_base4(text_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()

    class JRSModel(BertPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            classification_output = outputs[1]
            logits = self.classifier(classification_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 192
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), ), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_base/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained4/roberta_base4/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask).view(-1)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    ranks = toxic_pred.argsort().argsort()
    print(toxic_pred[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_large4(text_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, BertPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze()

    class JRSModel(BertPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 1)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask)
            classification_output = outputs[1]
            logits = self.classifier(classification_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 192
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), ), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_large/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained4/roberta_large4/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask).view(-1)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    ranks = toxic_pred.argsort().argsort()
    print(toxic_pred[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_base4(text_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 1)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 192
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), ), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_base/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained4/deberta_base4/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids).view(-1)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    ranks = toxic_pred.argsort().argsort()
    print(toxic_pred[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_large4(text_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 1)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 192
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), ), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_large/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained4/deberta_large4/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids).view(-1)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    ranks = toxic_pred.argsort().argsort()
    print(toxic_pred[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_base3(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 7)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), 7), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_base/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained2/roberta_base3/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(7):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_large3(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 7)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 7), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_large/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained2/roberta_large3/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(7):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_base3(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 7)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state'][:,0,:]
            logits = self.classifier(outputs)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 7), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_base/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained2/deberta_base3/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(7):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_base2(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_base/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained3/roberta_base2/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_large2(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_large/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained3/roberta_large2/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_base2(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 384
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_base/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained3/deberta_base2/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_large2(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 384
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_large/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained3/deberta_large2/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_base1(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 8

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_base/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained1/roberta_base1/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_roberta_large1(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import RobertaModel, RobertaPreTrainedModel, RobertaConfig, get_linear_schedule_with_warmup, RobertaTokenizerFast

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_attention_mask=True,
                                       return_token_type_ids=True,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(RobertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.roberta = RobertaModel(config)
            self.classifier = nn.Linear(config.hidden_size, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.roberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)['last_hidden_state']
            embeddings = torch.mean(outputs, axis=1)
            logits = self.classifier(embeddings)
            return logits

    start_time = time.time()

    # parameters
    max_len = 256
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/roberta_large/"
    config = RobertaConfig.from_pretrained(model_path+'config.json')
    config.hidden_dropout_prob = 0
    config.attention_probs_dropout_prob = 0
    tokenizer = RobertaTokenizerFast.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained1/roberta_large1/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_base1(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 384
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_base/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained1/deberta_base1/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks

def get_deberta_large1(text_list, weight_list):
    import numpy as np
    import pandas as pd
    import os
    from tqdm import tqdm
    import torch.nn as nn
    from torch import optim
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    import torch
    from torch.cuda.amp import autocast, GradScaler
    import time
    from transformers import DebertaModel, DebertaPreTrainedModel, DebertaConfig, get_linear_schedule_with_warmup, DebertaTokenizer
    from transformers.models.deberta.modeling_deberta import ContextPooler

    class JRSDataset(Dataset):
        def __init__(self, text_list, tokenizer, max_len):
            self.text_list=text_list
            self.tokenizer=tokenizer
            self.max_len=max_len
        def __len__(self):
            return len(self.text_list)
        def __getitem__(self, index):
            tokenized = self.tokenizer(text=self.text_list[index],
                                       padding='max_length',
                                       truncation=True,
                                       max_length=self.max_len,
                                       return_tensors='pt')
            return tokenized['input_ids'].squeeze(), tokenized['attention_mask'].squeeze(), tokenized['token_type_ids'].squeeze()

    class JRSModel(DebertaPreTrainedModel):
        def __init__(self, config):
            super(JRSModel, self).__init__(config)
            self.deberta = DebertaModel(config)
            self.pooler = ContextPooler(config)
            output_dim = self.pooler.output_dim
            self.classifier = nn.Linear(output_dim, 6)
            self.init_weights()
        def forward(self, input_ids, attention_mask=None, token_type_ids=None):
            outputs = self.deberta(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            encoder_layer = outputs[0]
            pooled_output = self.pooler(encoder_layer)
            logits = self.classifier(pooled_output)
            return logits

    start_time = time.time()

    # parameters
    max_len = 384
    batch_size = 4

    # build model
    toxic_pred = np.zeros((len(text_list), 6), dtype=np.float32)
    
    model_path = "../input/jrconfigs/deberta_large/"
    config = DebertaConfig.from_pretrained(model_path+'config.json')
    tokenizer = DebertaTokenizer.from_pretrained(model_path)
    model = JRSModel.from_pretrained('../input/jrpretrained1/deberta_large1/weights', config=config)
    model = model.cuda()
    model.eval()

    # iterator for validation
    dataset = JRSDataset(text_list, tokenizer, max_len)
    generator = DataLoader(dataset=dataset, 
                           batch_size=batch_size, 
                           shuffle=False,
                           num_workers=2, 
                           pin_memory=False)

    for j, (batch_input_ids, batch_attention_mask, batch_token_type_ids) in enumerate(generator):
        with torch.no_grad():
            start = j*batch_size
            end = start+batch_size
            if j == len(generator)-1:
                end = len(generator.dataset)
            batch_input_ids = batch_input_ids.cuda()
            batch_attention_mask = batch_attention_mask.cuda()
            batch_token_type_ids = batch_token_type_ids.cuda()
            with autocast():
                logits = model(batch_input_ids, batch_attention_mask, batch_token_type_ids)
            toxic_pred[start:end] = logits.sigmoid().cpu().data.numpy()

    ### result
    toxic_score = np.zeros((len(text_list), ), dtype=np.float32)
    for i in range(6):
        toxic_score += toxic_pred[:,i]*weight_list[i]
    ranks = toxic_score.argsort().argsort()
    print(toxic_score[:20])
    print(ranks[:20])

    end_time = time.time()
    print(end_time-start_time)

    return ranks


import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

df = pd.read_csv('../input/jigsaw-toxic-severity-rating/comments_to_score.csv')
id_list = df['comment_id'].values
text_list = df['text'].values

ranks13 = get_roberta_base4(text_list)
ranks14 = get_roberta_large4(text_list)
ranks15 = get_deberta_base4(text_list)
ranks16 = get_deberta_large4(text_list)

#weight_list9 = [0.0, 20.0, 7.0, 20.0, 10.0, 13.0, 13.0]
#ranks9 = get_roberta_base3(text_list, weight_list9)

#weight_list10 = [20.0, 8.0, 8.0, 13.0, 3.0, 12.0, 6.0]
#ranks10 = get_roberta_large3(text_list, weight_list10)

#weight_list11 = [3.0, 19.0, 2.0, 1.0, 2.0, 5.0, 5.0]
#ranks11 = get_deberta_base3(text_list, weight_list11)

##weight_list12 = [0.0, 4.0, 1.0, 1.0, 0.0, 1.0]
##ranks12 = get_deberta_large3(text_list, weight_list12)

#weight_list5 = [4.0, 2.0, 2.0, 0.0, 1.0, 2.0]
#ranks5 = get_roberta_base2(text_list, weight_list5)

#weight_list6 = [0.0, 4.0, 2.0, 1.0, 1.0, 2.0]
#ranks6 = get_roberta_large2(text_list, weight_list6)

#weight_list7 = [1.0, 4.0, 4.0, 2.0, 1.0, 2.0]
#ranks7 = get_deberta_base2(text_list, weight_list7)

#weight_list8 = [0.0, 4.0, 1.0, 1.0, 0.0, 1.0]
#ranks8 = get_deberta_large2(text_list, weight_list8)

#weight_list1 = [5.0, 18.0, 8.0, 3.0, 4.0, 10.0]
#ranks1 = get_roberta_base1(text_list, weight_list1)

#weight_list2 = [0.0, 20.0, 8.0, 4.0, 2.0, 15.0]
#ranks2 = get_roberta_large1(text_list, weight_list2)

#weight_list3 = [20.0, 18.0, 4.0, 4.0, 1.0, 4.0]
#ranks3 = get_deberta_base1(text_list, weight_list3)

#weight_list4 = [0.0, 15.0, 1.0, 3.0, 2.0, 2.0]
#ranks4 = get_deberta_large1(text_list, weight_list4)

#ranks = 0.5*((ranks1+ranks2+ranks3+ranks4+ranks5+ranks6+ranks7+ranks8)/8.0) + 0.25*((ranks9+ranks10+ranks11)/3.0) + 0.25*((ranks13+ranks14+ranks15+ranks16)/4.0)
ranks = (ranks13+ranks14+ranks15+ranks16)/4.0
print(ranks[:20])

ranks_breaktie = ranks.argsort().argsort()
print(ranks_breaktie[:20])

sub_df = pd.DataFrame(data={'comment_id': id_list, 'score': ranks_breaktie})
sub_df.to_csv('submission.csv', index=False)
