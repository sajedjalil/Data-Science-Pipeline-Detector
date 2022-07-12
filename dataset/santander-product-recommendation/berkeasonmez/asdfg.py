import csv
import datetime
import random
from operator import sub
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn import preprocessing, ensemble


def DetectBadValues(out,data):
    array = []
    value = 0
    for x in out:
        if data[x].strip() in ['', 'NA']:
            value = 0
        else:
            value = int(float(data[x]))
        array.append(value)

    return array

def ReplaceValues(dict_,data, lab):
    value = data[lab].strip()
    if value not in ['','NA']:
        num = dict_[lab][value]
    else:
        num = dict_[lab][-99]
    return num

def FindAge(data):
    ageMean = 40.
    ageMin = 18.
    ageMax = 100.
    diff = ageMax - ageMin
    age = data['age'].strip()
    if age == 'NA' or age == '':
        age = ageMean
    else:
        age = float(age)
        if age < ageMin:
            age = ageMin
        elif age > ageMax:
            age = ageMax
    return round( (age - ageMin) / diff, 4)

def FindAntiguedad(data):
    valueMin = 0.
    valueMax = 256.
    diff = valueMax - valueMin
    anti = data['antiguedad'].strip()
    if anti == 'NA' or anti == '':
        anti = valueMin
    else:
        anti = float(anti)
        if anti < valueMin:
            anti = valueMin
        elif anti > valueMax:
            anti = valueMax
    return round((anti-valueMin) / diff, 4)

def FindRent(data):
    rentMin = 0.
    rentMax = 1500000.
    diff = rentMax-rentMin
    rentAverage = {'ALBACETE': 76895,  'ALICANTE': 60562,  'ALMERIA': 77815,  'ASTURIAS': 83995,  'AVILA': 78525,  'BADAJOZ': 60155,  'BALEARS, ILLES': 114223,  'BARCELONA': 135149,  'BURGOS': 87410, 'NAVARRA' : 101850,
    'CACERES': 78691,  'CADIZ': 75397,  'CANTABRIA': 87142,  'CASTELLON': 70359,  'CEUTA': 333283, 'CIUDAD REAL': 61962,  'CORDOBA': 63260,  'CORUÑA, A': 103567,  'CUENCA': 70751,  'GIRONA': 100208,  'GRANADA': 80489,
    'GUADALAJARA': 100635,  'HUELVA': 75534,  'HUESCA': 80324,  'JAEN': 67016,  'LEON': 76339,  'LERIDA': 59191,  'LUGO': 68219,  'MADRID': 141381,  'MALAGA': 89534,  'MELILLA': 116469, 'GIPUZKOA': 101850,
    'MURCIA': 68713,  'OURENSE': 78776,  'PALENCIA': 90843,  'PALMAS, LAS': 78168,  'PONTEVEDRA': 94328,  'RIOJA, LA': 91545,  'SALAMANCA': 88738,  'SANTA CRUZ DE TENERIFE': 83383, 'ALAVA': 101850, 'BIZKAIA' : 101850,
    'SEGOVIA': 81287,  'SEVILLA': 94814,  'SORIA': 71615,  'TARRAGONA': 81330,  'TERUEL': 64053,  'TOLEDO': 65242,  'UNKNOWN': 103689,  'VALENCIA': 73463,  'VALLADOLID': 92032,  'ZAMORA': 73727,  'ZARAGOZA': 98827}

    #missing_value = 101850.
    rent = data['renta'].strip()
    city = data['nomprov']
    if rent == 'NA' or rent == '':
        if city== 'NA' or city == '':
            rent = float(rentAverage['UNKNOWN'])
        else:
            rent = float(rentAverage[city])
    else:
        rent = float(rent)
        if rent < rentMin:
            rent = rentMin
        elif rent > rentMax:
            rent = rentMax

    return round((rent-rentMin) / diff, 6)


def FindMonth(data):
    return int(data['fecha_dato'].split('-')[1])

def FindMonth2(data):
    if data['fecha_alta'].strip() == 'NA' or data['fecha_alta'].strip() == '':
        return int(random.choice([1,2,3,4,5,6,7,8,9,10,11,12]))
    else:
        return int(data['fecha_alta'].split('-')[1])
    
def dig_data(dict_,in_file_name, clients, clients_2):
    
    feature_list = list(dict_.keys())
    output_list = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']
    dates=['2015-04-28', '2015-05-28', '2015-06-28', '2016-04-28', '2016-05-28', '2016-06-28']
    a_list = []
    b_list = []
        
    for row in csv.DictReader(in_file_name):
        
        fecha_date=row['fecha_dato']
        
        if fecha_date in dates:
            
            client_id = int(row['ncodpers'])
        
            if fecha_date[6]=='4':  
                clients_2[client_id] =  DetectBadValues(output_list,row)
             
        
            elif fecha_date[6]=='5':
                clients[client_id] =  DetectBadValues(output_list,row)
             
            
            else:               
                new_list = []
                for col in feature_list:
                    new_list.append( ReplaceValues(dictionary,row, col) )
                    
                
                new_list.append(FindAge(row))
                new_list.append(FindMonth(row))
                new_list.append(FindMonth2(row))
                new_list.append(FindAntiguedad(row)) 
                new_list.append(FindRent(row))
                
                if fecha_date == '2016-06-28':
                    a_list.append(new_list + clients.get(client_id, [0]*22) + clients_2.get(client_id, [0]*22))
                    
                else:                   
                    c_list = clients.get(client_id, [0]*22)                          
                    goods = [max(a-b,0) for (a,b) in zip(DetectBadValues(output_list,row), c_list)]
                    if sum(goods) > 0:
                        for ind, prod in enumerate(goods):
                            if prod>0:
                                a_list.append(new_list+c_list+clients_2.get(client_id, [0]*22))
                                b_list.append(ind)
        
        
    return a_list, b_list, clients, clients_2
    

def runXGB(training_x, training_y, seed_val):
    param = {}
    param['objective'] = 'multi:softprob'
    param['eta'] = 0.1
    param['max_depth'] = 8
    param['silent'] = 1
    param['num_class'] = 22
    param['eval_metric'] = "mlogloss"
    param['min_child_weight'] = 10
    param['subsample'] = 0.8
    param['colsample_bytree'] = 0.8
    param['seed'] = seed_val

    xgtrain = xgb.DMatrix(training_x, label=training_y)
    model = xgb.train(list(param.items()), xgtrain, 75)
    return model


dictionary = {'ind_empleado'  : {-99:0, 'N':1, 'B':2, 'F':3, 'A':4, 'S':5},
'sexo'          : {'V':0, 'H':1, -99:2},
'ind_nuevo'     : {'0':0, '1':1, -99:1},
'indrel'        : {'1':0, '99':1, -99:1},
'indrel_1mes'   : {-99:0, '1.0':1, '1':1, '2.0':2, '2':2, '3.0':3, '3':3, '4.0':4, '4':4, 'P':5},
'tiprel_1mes'   : {-99:0, 'I':1, 'A':2, 'P':3, 'R':4, 'N':5},
'indresi'       : {-99:0, 'S':1, 'N':2},
'indext'        : {-99:0, 'S':1, 'N':2},
#'conyuemp'      : {-99:0, 'S':1, 'N':2},
'indfall'       : {-99:0, 'S':1, 'N':2},
#'tipodom'       : {-99:0, '1':1},
'ind_actividad_cliente' : {'0':0, '1':1, -99:2},
'segmento'      : {'02 - PARTICULARES':0, '03 - UNIVERSITARIO':1, '01 - TOP':2, -99:3},
'pais_residencia' : {'LV': 102, 'BE': 12, 'BG': 50, 'BA': 61, 'BM': 117, 'BO': 62, 'JP': 82, 'JM': 116, 'BR': 17, 'BY': 64, 'BZ': 113, 'RU': 43, 'RS': 89, 'RO': 41, 'GW': 99, 'GT': 44, 'GR': 39, 'GQ': 73, 'GE': 78, 'GB': 9, 'GA': 45, 'GN': 98, 'GM': 110, 'GI': 96, 'GH': 88, 'OM': 100, 'HR': 67, 'HU': 106, 'HK': 34, 'HN': 22, 'AD': 35, 'PR': 40, 'PT': 26, 'PY': 51, 'PA': 60, 'PE': 20, 'PK': 84, 'PH': 91, 'PL': 30, 'EE': 52, 'EG': 74, 'ZA': 75, 'EC': 19, 'AL': 25, 'VN': 90, 'ET': 54, 'ZW': 114, 'ES': 0, 'MD': 68, 'UY': 77, 'MM': 94, 'ML': 104, 'US': 15, 'MT': 118, 'MR': 48, 'UA': 49, 'MX': 16, 'IL': 42, 'FR': 8, 'MA': 38, 'FI': 23, 'NI': 33, 'NL': 7, 'NO': 46, 'NG': 83, 'NZ': 93, 'CI': 57, 'CH': 3, 'CO': 21, 'CN': 28, 'CM': 55, 'CL': 4, 'CA': 2, 'CG': 101, 'CF': 109, 'CD': 112, 'CZ': 36, 'CR': 32, 'CU': 72, 'KE': 65, 'KH': 95, 'SV': 53, 'SK': 69, 'KR': 87, 'KW': 92, 'SN': 47, 'SL': 97, 'KZ': 111, 'SA': 56, 'SG': 66, 'SE': 24, 'DO': 11, 'DJ': 115, 'DK': 76, 'DE': 10, 'DZ': 80, 'MK': 105, -99: 1, 'LB': 81, 'TW': 29, 'TR': 70, 'TN': 85, 'LT': 103, 'LU': 59, 'TH': 79, 'TG': 86, 'LY': 108, 'AE': 37, 'VE': 14, 'IS': 107, 'IT': 18, 'AO': 71, 'AR': 13, 'AU': 63, 'AT': 6, 'IN': 31, 'IE': 5, 'QA': 58, 'MZ': 27},
'canal_entrada' : {'013': 49, 'KHP': 160, 'KHQ': 157, 'KHR': 161, 'KHS': 162, 'KHK': 10, 'KHL': 0, 'KHM': 12, 'KHN': 21, 'KHO': 13, 'KHA': 22, 'KHC': 9, 'KHD': 2, 'KHE': 1, 'KHF': 19, '025': 159, 'KAC': 57, 'KAB': 28, 'KAA': 39, 'KAG': 26, 'KAF': 23, 'KAE': 30, 'KAD': 16, 'KAK': 51, 'KAJ': 41, 'KAI': 35, 'KAH': 31, 'KAO': 94, 'KAN': 110, 'KAM': 107, 'KAL': 74, 'KAS': 70, 'KAR': 32, 'KAQ': 37, 'KAP': 46, 'KAW': 76, 'KAV': 139, 'KAU': 142, 'KAT': 5, 'KAZ': 7, 'KAY': 54, 'KBJ': 133, 'KBH': 90, 'KBN': 122, 'KBO': 64, 'KBL': 88, 'KBM': 135, 'KBB': 131, 'KBF': 102, 'KBG': 17, 'KBD': 109, 'KBE': 119, 'KBZ': 67, 'KBX': 116, 'KBY': 111, 'KBR': 101, 'KBS': 118, 'KBP': 121, 'KBQ': 62, 'KBV': 100, 'KBW': 114, 'KBU': 55, 'KCE': 86, 'KCD': 85, 'KCG': 59, 'KCF': 105, 'KCA': 73, 'KCC': 29, 'KCB': 78, 'KCM': 82, 'KCL': 53, 'KCO': 104, 'KCN': 81, 'KCI': 65, 'KCH': 84, 'KCK': 52, 'KCJ': 156, 'KCU': 115, 'KCT': 112, 'KCV': 106, 'KCQ': 154, 'KCP': 129, 'KCS': 77, 'KCR': 153, 'KCX': 120, 'RED': 8, 'KDL': 158, 'KDM': 130, 'KDN': 151, 'KDO': 60, 'KDH': 14, 'KDI': 150, 'KDD': 113, 'KDE': 47, 'KDF': 127, 'KDG': 126, 'KDA': 63, 'KDB': 117, 'KDC': 75, 'KDX': 69, 'KDY': 61, 'KDZ': 99, 'KDT': 58, 'KDU': 79, 'KDV': 91, 'KDW': 132, 'KDP': 103, 'KDQ': 80, 'KDR': 56, 'KDS': 124, 'K00': 50, 'KEO': 96, 'KEN': 137, 'KEM': 155, 'KEL': 125, 'KEK': 145, 'KEJ': 95, 'KEI': 97, 'KEH': 15, 'KEG': 136, 'KEF': 128, 'KEE': 152, 'KED': 143, 'KEC': 66, 'KEB': 123, 'KEA': 89, 'KEZ': 108, 'KEY': 93, 'KEW': 98, 'KEV': 87, 'KEU': 72, 'KES': 68, 'KEQ': 138, -99: 6, 'KFV': 48, 'KFT': 92, 'KFU': 36, 'KFR': 144, 'KFS': 38, 'KFP': 40, 'KFF': 45, 'KFG': 27, 'KFD': 25, 'KFE': 148, 'KFB': 146, 'KFC': 4, 'KFA': 3, 'KFN': 42, 'KFL': 34, 'KFM': 141, 'KFJ': 33, 'KFK': 20, 'KFH': 140, 'KFI': 134, '007': 71, '004': 83, 'KGU': 149, 'KGW': 147, 'KGV': 43, 'KGY': 44, 'KGX': 24, 'KGC': 18, 'KGN': 11}
}
output_list = ['ind_cco_fin_ult1','ind_cder_fin_ult1','ind_cno_fin_ult1','ind_ctju_fin_ult1','ind_ctma_fin_ult1','ind_ctop_fin_ult1','ind_ctpp_fin_ult1','ind_deco_fin_ult1','ind_deme_fin_ult1','ind_dela_fin_ult1','ind_ecue_fin_ult1','ind_fond_fin_ult1','ind_hip_fin_ult1','ind_plan_fin_ult1','ind_pres_fin_ult1','ind_reca_fin_ult1','ind_tjcr_fin_ult1','ind_valo_fin_ult1','ind_viv_fin_ult1','ind_nomina_ult1','ind_nom_pens_ult1','ind_recibo_ult1']

training_data =  open("../input/train_ver2.csv")
test_data = open("../input/test_ver2.csv")    
    
a_list_train, b_list_train, clients, clients_2 = dig_data(dictionary,training_data, {}, {})    
a_list_test, b_list_test, clients, clients_2 = dig_data(dictionary,test_data, clients, clients_2)

training_data.close()
test_data.close()
    
model = runXGB(np.array(a_list_train), np.array(b_list_train), seed_val=0)
predictions = model.predict(xgb.DMatrix(np.array(a_list_test)))

test_id = np.array(pd.read_csv("../input/test_ver2.csv", usecols=['ncodpers'])['ncodpers'])
goods = []
    
for i, k in enumerate(test_id):
    goods.append([max(a-b,0) for (a,b) in zip(predictions[i,:], clients[k])])
              
target_cols = np.array(output_list)
predictions = np.argsort(np.array(goods), axis=1)
predictions = np.fliplr(predictions)[:,:7]
final_predictions = [" ".join(list(target_cols[pred])) for pred in predictions]
pd.DataFrame({'ncodpers':test_id, 'added_products':final_predictions}).to_csv('sub_xgb_new_5.csv', index=False)
    
print('THE END')