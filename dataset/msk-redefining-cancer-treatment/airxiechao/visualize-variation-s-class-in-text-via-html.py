import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re

# load data
def load_train_data():
    train_class = pd.read_csv('../input/training_variants')
    
    train_text = []
    with open('../input/training_text', 'r', encoding='utf-8') as f:
        f.readline()
        lines = f.readlines();
        for line in lines:
            ID, Text = line.split('||')
            train_text.append([int(ID), Text])
            
        train_text = pd.DataFrame(data=train_text, columns=['ID', 'Text'])
        
    train = pd.merge(train_class, train_text, on='ID')
    
    return train


def load_test_data():
    test_class = pd.read_csv('../input/test_variants')
    
    test_text = []
    with open('../input/test_text', 'r', encoding='utf-8') as f:
        f.readline()
        lines = f.readlines();
        for line in lines:
            ID, Text = line.split('||')
            test_text.append([int(ID), Text])
            
        test_text = pd.DataFrame(data=test_text, columns=['ID', 'Text'])

    test = pd.merge(test_class, test_text, on='ID')
    
    return test
    

train = load_train_data()
test = load_test_data()

dict_variation_class_train = train[['Variation', 'Class']].set_index('Variation').to_dict()['Class']


# annotate variations in text
def replace_variation_to_spans(text):
  #find all variations in text
  tokens = set(re.split('\W+', text))
  
  #annotate train variations
  keys = set(dict_variation_class_train.keys())
  variations_in_text = tokens.intersection(keys)
  for v in variations_in_text:
    cls = dict_variation_class_train[v]
    text = re.sub('('+v+')', '<span class="y y_'+str(cls)+'">\\1</span>', text)
  
  text = text.strip()
  lines = text.split('. ')
  lines = '\n'.join(['<p>'+line+'</p>' for line in lines])
  return lines

# visualize text via HTML
def text_to_html(df, idx_text):
  with open('./'+str(idx_text)+'.html', 'w', encoding='utf-8') as f:
    template_1 = '''
      <!DOCTYPE html>
      <html>
      <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width,initial-scale=1.0,user-scalable=no" />
        <title></title>
      
        <script type="text/javascript" src="http://code.jquery.com/jquery-3.2.1.min.js"></script>
        <script type="text/javascript"></script>
        
        <style type="text/css">
          html, body {
            position: relative;
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            background-color: #fff;
            font-family: sans-serif;
          }
          
          .y{
            border: 1px solid red;
            position: relative;
            color: red;
            font-weight: bold;
          }
          
          .y:after{
            position: absolute;
            top: -15px;
            left: 50%;
            font-size: 14px;
          }
          
          .y_1:after{ content: '1'; }
          .y_2:after{ content: '2'; }
          .y_3:after{ content: '3'; }
          .y_4:after{ content: '4'; }
          .y_5:after{ content: '5'; }
          .y_6:after{ content: '6'; }
          .y_7:after{ content: '7'; }
          .y_8:after{ content: '8'; }
          .y_9:after{ content: '9'; }
            
        </style>
          
      </head>
      <body>
        <div class="text">
    '''
    
    template_2 = '''
        </div>
      </body>
      </html>
    '''
    f.write(template_1)
    f.write(replace_variation_to_spans(df.Text[idx_text]))
    f.write(template_2)
  
  
text_to_html(train, 2)

