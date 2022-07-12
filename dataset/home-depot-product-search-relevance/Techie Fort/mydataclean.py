# -*- coding: ISO-8859-1 -*-
#### Muhammd Nawaz

import numpy as np
import pandas as pd
from nltk.stem.porter import *
stemmer = PorterStemmer()
import re

strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}
def str_stemmer(s):
    s = str(s)
    if isinstance(s, str):
        s = s.lower()
        s = s.replace("toliet", "toilet")
        s = s.replace("airconditioner", "air condition")
        s = s.replace("vinal", "vinyl")
        s = s.replace("vynal", "vinyl")
        s = s.replace("skill", "skil")
        s = s.replace("snowbl", "snow bl")
        s = s.replace("plexigla", "plexi gla")
        s = s.replace("rustoleum", "rust oleum")
        s = s.replace("whirpool", "whirlpool")
        s = s.replace("whirlpoolga", "whirlpool ga")
        s = s.replace("whirlpoolstainless", "whirlpool stainless")
        ##########################
        s = re.sub("(^|\s+)a c($|\s+)", r" air conditioner  ", s)  # me
        s = re.sub("(^|\s+)a/c($|\s+)", r" air conditioner  ", s)  # me
        s = s.replace("rumbl;estone", "rumblestone")
        s = s.replace("6' weather shield", "6' weathershield")
        s = s.replace("footaluminum", "foot aluminum")
        s = s.replace(" j chann", " j-channel")
        s = s.replace("bi-fold", "bifold")
        s = s.replace("3 in one", "3-in-one")
        s = s.replace(" l ", " 90 degree ")
        s = s.replace("pl-c ", "plc ")
        s = s.replace("l-angle", "90 degree")
        s = s.replace("shelving", "shelf")
        s = s.replace("shelves", "shelf")
        s = s.replace("shelve", "shelf")
        s = s.replace("(", " ")
        s = s.replace(")", " ")
        s = s.replace("rost oluem", "rustoleum")
        s = s.replace("rost-oluem", "rustoleum")
        s = s.replace("vent-free", "ventfree")
        s = s.replace("ventless", "ventfree")
        s = s.replace("vent less", "ventfree")
        s = s.replace("vent free", "ventfree")
        s = s.replace("doors", " door ")
#        s = s.replace("heater", "heat")
        s = s.replace("-gph", " gph")
        s = s.replace(" sun ", " solar ")
        s = s.replace("compressro", "compressor")
        s = s.replace("windoww", "window")
        s = s.replace("claissic", "classic")
        s = s.replace("sander", "sand")
        s = s.replace("5 induct", "5 in. duct")
        s = s.replace("-light", " light")
        #s = s.replace("hi strength", " ")
        s = s.replace(" ddor", " door ")
        s = s.replace(" dim able", " dimable")
        s = s.replace(" dog ", " pet ")
        s = s.replace(".- ", ". ")
        s = s.replace(" plna ", " plan ")
        s = s.replace("bluwood", "blurwood")
        s = s.replace(" elboq", " elbow")
        s = s.replace("cats ", "pets ")
        s = s.replace("chase lounge", "chaise lounge")
        s = s.replace("charbroi l", "char broil")
        s = s.replace("charbroil", "char broil")
        s = s.replace("chibli", "chili")
        s = s.replace("chissel", "chisel")
        s = s.replace("closet maid", "closetmaid")
        s = s.replace("closet maiid", "closetmaid")
        s = s.replace("closetmade", "closetmaid")
        s = s.replace("caseing", "casing")
        s = s.replace("enchors", "anchor s")
        s = s.replace("concrete bled", "concrete blend")
        s = s.replace("rust o leum", "rust oleum")
        s = s.replace("concrete saw", "concrete saw blade cutter")
        s = s.replace("congolium", "congoleum")
        s = s.replace("connectorv", "connector")
        s = s.replace("convction", "convection")
        s = s.replace("aeretor", "aerator")
        s = s.replace("corsar", "coarse")
        s = s.replace("costum", "Custom")
        s = s.replace("couchen", "couch cushion")
        s = s.replace("freezer", "refrigerator")
        s = s.replace("counterdepth", "counter depth")
        s = s.replace("jigsaw", "jig saw")
        s = s.replace(" of ", " ")
        s = s.replace(" a ", " ")
        s = s.replace("cranner", "crane")
        s = s.replace("cupboards", "cabinet")
        s = s.replace("hardiebacker", "hardie backer")
        s = s.replace("cutoff", "cut off")
        s = s.replace("dal tile ", "daltile ")
        s = s.replace("dampiner", "dampener")
        s = s.replace("daylillies", "Daylily daylillies")
        s = s.replace("decostrip", "deco strip")
        s = s.replace("dehumidifyer", "dehumidifier")
        s = s.replace("delata", "delta")
        s = s.replace("ranshower", "ran shower")
        s = s.replace("empact", "e impact")
        s = s.replace("hammerdrill", "hammer drill")
        s = s.replace("sprinkler", "spray")
        s = s.replace("diining", "dining")
        s = s.replace("dleta", "delta")
        s = s.replace("doggie", "dog")
        s = s.replace("doggie", "dog")
        s = s.replace("knobsw", "knob sw")
        s = s.replace("latch uard", "latch guard")
        s = s.replace("stopper", "stop")
        s = s.replace("doorbell", "door bell")
        s = s.replace("spoutts", "spout")
        s = s.replace("drainage", "drain")
        s = s.replace("dream line", "dreamline")
        s = s.replace("ppanel", "panel")
        s = s.replace("airconditioner", "air conditioner")
        s = s.replace("desighn", "design")
        s = s.replace("el 90deg hxh", "90 degree hxh")
        s = s.replace("elrctric", "electric")
        s = s.replace("exteriordoors", "exterior doors")
        s = s.replace("lumber", "timber")
        s = s.replace("fi nish screws", "finish screws")
        s = s.replace("matt ", "mat ")
        s = s.replace(" shoer ", " shower ")
        s = s.replace("frog tape", "frogtape")
        s = s.replace("furnance", "furnace")
        s = s.replace("galvanisedround", "galvanised round")
        s = s.replace("geotextile ", "geotextile fabric")
        s = s.replace("glue", "adhesive")
        s = s.replace("greased lightning", "greased lightning degreaser")
        s = s.replace("gwtters", "gutters")
        s = s.replace("hacksaw", "hack saw")
        s = s.replace("headlamps", "headlamps led")
        s = s.replace("dutty", "duty")
        s = s.replace("holiday", "holiday christmas")
        s = s.replace("lightshow", "light show")
        s = s.replace("cinamon", "cinnamon")
        s = s.replace("hoovwe", "hoover")
        s = s.replace("huskvarna", "husqvarna")
        s = s.replace("i/2", "1/2")
        s = s.replace("ibeam", "beam")
        s = s.replace("icycle", "icicle")
        s = s.replace("inferni", "inferno")
        s = s.replace("insullation", "insulation")
        s = s.replace("jeldwen", "jeld wen")
        s = s.replace("_", " ")
        s = s.replace("jigsaww", "jigsaw")
        s = s.replace("jig saw", "jigsaw")
        s = s.replace("jimmyproof", "jimmy proof")
        s = s.replace("kholerhighland", "kholer highland")
        s = s.replace("kitchenaid", "kitchen aid")
        s = s.replace("layndry", "laundry")
        s = s.replace("linesman", "lineman")
        s = s.replace("snowblower", "snow blower")
        s = s.replace("pfistersaxton", "pfister saxton")
        s = s.replace("rustolem", "rustoleum")
        s = s.replace("rybi", "ryobi")
        s = s.replace("samsungelectric", "samsung electric")
        s = s.replace("saw zall", "sawzall")
        s = s.replace("ligghts", "lights")
        s = s.replace("vynal", "vinyl")
        s = s.replace("whitecomposite", "white composite")
        s = s.replace("woodflooring", "wood flooring")
        s = s.replace("zwave", "z wave")
        s = s.replace("anntenias", "antenna")
        s = s.replace("antenna pole", "antennapole")
        s = s.replace(" tiiles", " tiles")
        s = s.replace("patioo", "patio")
        s = s.replace("bandsaw", "band saw")
        s = s.replace("bathfan", "bath fan")
        s = s.replace("lawnmower", "lawn mower")
        s = s.replace("batteryfor ", "battery ")
        s = s.replace("bedbug", "bed bug")
        s = s.replace(" hawian ", " hawaiian ")
        s = s.replace("paint samples and names", "paint ")
        s = s.replace("ultrta", "ultra")
        s = s.replace("hard wood", "hardwood")
        s = s.replace("deck over", "deckover")
        s = s.replace("bidat", "bidet")
        s = s.replace("bigass", "big ass")
        s = s.replace("&", " ")
        s = s.replace("deckerbattery", "decker battery")
        s = s.replace("tapeexhaust", "tape exhaust")
        s = s.replace(" lazer ", " laser ")
        s = s.replace("tab;le", "table")
        s = s.replace(";", " ")
        s = s.replace("buggy beds", "buggybeds")
        s = s.replace("wjth", " ")
        s = s.replace("cablrail", "cabl rail")
        s = s.replace(" iii ", " 3 ")
        s = s.replace("centerpise", " ax centerpiece")
        s = s.replace("chain saw", " chainsaw ")
        s = s.replace("chair molding", "chair moulding")
        s = s.replace(" decke ", " decker ")
        s = s.replace("insulaation", "insulation")
        s = re.sub("(\d+)plinth", r"\1 plinth", s)
        s = s.replace("3 pac ", "3 pack ")
        s = s.replace("bbrush", "brush")
        s = s.replace("3 prong", "3-prong")
        s = re.sub("(\d+) tier", r"\1-tier", s)
        s = re.sub("(\d+)hp", r"\1 hp", s)
        s = re.sub("(\d+) pc ", r"\1 pc. ", s)
        s = re.sub("(\d+) pcs ", r"\1 pc. ", s)
        s = re.sub("(\d+)-piece ", r"\1 pc. ", s)
        s = re.sub("(\d+) piece ", r"\1 pc. ", s)
        s = re.sub("(\d+)-pieces ", r"\1 pc. ", s)
        s = re.sub("^t (\d+) ", r"t\1 ", s)
        s = s.replace("blace", "black")
        s = s.replace("horsepower", "hp")
        s = s.replace("vlve", "valve")
        s = s.replace("weatherstrip", "weather strip")
        s = s.replace("3/4plywood", "3/4 plywood")
        s = s.replace("couipling", "coupling")
        s = s.replace("entension", "extension")
        s = s.replace("cubic", "cu.")
        s = s.replace("litr", "lite")
        s = s.replace("refigrator", "refrigerator")
        s = s.replace("par30l", "30 par")
        s = s.replace("cadetcordless", "cadet cordless")
        s = s.replace("cans", "can")
        s = s.replace("doooors", "door")
        s = s.replace("copoktop", "cooktop")
        s = s.replace("dutchlap", "dutch lap")
        s = s.replace(" an ", " ")
        s = s.replace(" with ", " ")
        s = s.replace("sillcick", "sillcock")
        s = s.replace("pos cap", "post cap")
        s = s.replace("ceing", "ceiling")
        s = s.replace(" wife", " wire")
        s = s.replace("lightbulb", "light bulb")
        s = s.replace("duking", "decking")
        s = s.replace(" type ", " ")
        s = s.replace("bolt", "screw")
        s = s.replace("gasoline", "gas")
        s = s.replace("showerstall", "shower stall")
        s = s.replace("18-ga brad", "18-gauge brad")
        s = s.replace("90-degree", "90 degree")
        s = s.replace("refrierator", "refrigerator")
        s = s.replace(" end ", " ")
        s = s.replace(" facia ", " fascia ")
        s = s.replace("-gauge", " gauge")
        s = s.replace("comercialcarpet", "comercial carpet")
        s = s.replace("bookshelf", "book shelf")
        s = s.replace("mivrowave", "microwave")
        s = s.replace("padspads", "pad")
        s = s.replace("9-lite", "9 lite")
        s = s.replace("refrigeratorators", "refrigerator")
        s = s.replace("fridges", "refrigerator")
        s = s.replace("fridge", "refrigerator")
        s = s.replace(" peen ", " pein ")
        s = s.replace("sch40", "sch. 40")
        s = s.replace("3 way", "3-way")
        s = re.sub("pvc ([t])(\s|$)", r"pvc tee", s)
        s = s.replace("1/4 od 1/4", "1/4 o.d. 1/4")
        s = s.replace(" blde", " blade")
        s = s.replace(" elec ", " electrical ")
        s = s.replace(" elec.", " electrical ")
        s = s.replace("100 00 btu", "100,00 btu")
        s = s.replace("10000", "100,00 btu")
        s = s.replace("-packs", " pack")
        s = s.replace("-pack", " pack")
        s = s.replace("-peice", " pecice")
        s = s.replace(" qt ", " qt. ")
        s = s.replace(" by ", " xby ")
        s = s.replace(" s. p ", " single-pole ")
        s = s.replace("5 fj ", "5 finger-joint ")
        s = s.replace(" qt ", " qt. ")
        s = s.replace(" pvcp", " pvc ")
        s = s.replace(" 20threaded", " 20 threaded")
        s = s.replace(" s.s ", " metal ")
        s = s.replace("nuts/ ", "nuts per ")
        s = re.sub("(\d+)-qt\.", r"\1 qt.", s)
        s = s.replace(" zero ", " 0 ")
        s = s.replace(" zero-", " 0 ")
        s = s.replace("/ac", " air conditioner")
        s = s.replace(" ac ", r" air conditioner ")
        s = s.replace(" air cond ", r" air conditioner ")
        s = re.sub(" air con$", r" air conditioner ", s)
        s = s.replace("micro wave", "microwave")
        ##########################
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s)  # Split words with a.A
        s = s.lower()
        s = s.replace("  ", " ")
        s = re.sub(r"([0-9]),([0-9])", r"\1\2", s)
        s = s.replace(",", " ")
        s = s.replace("$", " ")
        s = s.replace("?", " ")
        s = s.replace("-", " ")
        s = s.replace("//", "/")
        s = s.replace("..", ".")
        s = s.replace(" / ", " ")
        s = s.replace(" \\ ", " ")
        s = s.replace(".", " . ")
        s = re.sub(r"(^\.|/)", r"", s)
        s = re.sub(r"(\.|/)$", r"", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = s.replace(" x ", " xby ")
        s = re.sub(r"([a-z])( *)\.( *)([a-z])", r"\1 \4", s)
        s = re.sub(r"([a-z])( *)/( *)([a-z])", r"\1 \4", s)
        s = s.replace("*", " xby ")
        s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
        s = re.sub(r"([0-9]+)( *)(inches|inch|in|'|inc|'x)(\.|\s+|$)", r"\1in. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)(\.|\s+|$)", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)(\.|\s+|$)", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')(\.|\s+|$)", r"\1ft. ", s)
        s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
        s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)(\.|\s+|$)", r"\1sq.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)(\.|\s+|$)", r"\1cu.ft. ", s)
        s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)(\.|\s+|$)", r"\1gal. ", s)
        s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)(\.|\s+|$)", r"\1oz. ", s)
        s = re.sub(r"([0-9]+)( *)(centimeters|cm)(\.|\s+|$)", r"\1cm. ", s)
        s = re.sub(r"([0-9]+)( *)(milimeters|mm)(\.|\s+|$)", r"\1mm. ", s)
        s = s.replace("Â°", " degrees ")
        s = re.sub(r"([0-9]+)( *)(degrees|degree)(\.|\s+|$)", r"\1deg. ", s)
        s = s.replace(" v ", " volts ")
        s = re.sub(r"([0-9]+)( *)(volts|volt|v)(\.|\s+|$)", r"\1volt. ", s)
        s = re.sub(r"(\s+|^)r(-|\s+)(\d+)(\s+|$)", r" r\3 ", s)
        s = re.sub(r"(\s+|^)t(-|\s+)(\d+)(\s+|$)", r" t\3 ", s)
        s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)(\.|\s+|$)", r"\1amp. ", s)
        s = re.sub(r"([0-9]+)( *)(watt|watts|w)(\.|\s+|$)", r"\1watt. ", s)
        s = s.replace("t 111", "t1 11")
        s = s.replace("  ", " ")
        s = s.replace(" . ", " ")
        s = (" ").join([str(strNum[z]) if z in strNum else z for z in s.split(" ")])
        s = (" ").join([stemmer.stem(z) for z in s.split(" ")])
        s = s.replace("ceadar", "cedar")
        return s
    else:
        return "null"

#Ref: Turing test kaggle
def spell_correction(s, automatic_spell_check_dict={}):
    s = str(s)
    s = s.lower()
    s = s.replace("craftsm,an","craftsman")        
    s = re.sub(r'depot.com/search=', '', s)
    s = re.sub(r'pilers,needlenose', 'pliers, needle nose', s)    
    
    s=s.replace("ttt","tt")    
    s=s.replace("lll","ll") 
    s=s.replace("nnn","nn") 
    s=s.replace("rrr","rr") 
    s=s.replace("sss","ss") 
    s=s.replace("zzz","zz")
    s=s.replace("ccc","cc")
    s=s.replace("eee","ee")
    
    s=s.replace("acccessories","accessories")
    s=re.sub(r'\bscott\b', 'scotts', s)
    s=re.sub(r'\borgainzer\b', 'organizer', s)
    s=re.sub(r'\bshark bite\b', 'sharkbite',s)
    
    s=s.replace("hinges with pishinges with pins","hinges with pins")    
    s=s.replace("virtue usa","virtu usa")
    s=re.sub('outdoor(?=[a-rt-z])', 'outdoor ', s)
    s=re.sub(r'\bdim able\b',"dimmable", s) 
    s=re.sub(r'\blink able\b',"linkable", s)
    s=re.sub(r'\bm aple\b',"maple", s)
    s=s.replace("aire acondicionado", "air conditioner")
    s=s.replace("borsh in dishwasher", "bosch dishwasher")
    s=re.sub(r'\bapt size\b','appartment size', s)
    s=re.sub(r'\barm[e|o]r max\b','armormax', s)
    s=re.sub(r' ss ',' stainless steel ', s)
    s=re.sub(r'\bmay tag\b','maytag', s)
    s=re.sub(r'\bback blash\b','backsplash', s)
    s=re.sub(r'\bbum boo\b','bamboo', s)
    s=re.sub(r'(?<=[0-9] )but\b','btu', s)
    s=re.sub(r'\bcharbroi l\b','charbroil', s)
    s=re.sub(r'\bair cond[it]*\b','air conditioner', s)
    s=re.sub(r'\bscrew conn\b','screw connector', s)
    s=re.sub(r'\bblack decker\b','black and decker', s)
    s=re.sub(r'\bchristmas din\b','christmas dinosaur', s)
    s=re.sub(r'\bdoug fir\b','douglas fir', s)
    s=re.sub(r'\belephant ear\b','elephant ears', s)
    s=re.sub(r'\bt emp gauge\b','temperature gauge', s)
    s=re.sub(r'\bsika felx\b','sikaflex', s)
    s=re.sub(r'\bsquare d\b', 'squared', s)
    s=re.sub(r'\bbehring\b', 'behr', s)
    s=re.sub(r'\bcam\b', 'camera', s)
    s=re.sub(r'\bjuke box\b', 'jukebox', s)
    s=re.sub(r'\brust o leum\b', 'rust oleum', s)
    s=re.sub(r'\bx mas\b', 'christmas', s)
    s=re.sub(r'\bmeld wen\b', 'jeld wen', s)
    s=re.sub(r'\bg e\b', 'ge', s)
    s=re.sub(r'\bmirr edge\b', 'mirredge', s)
    s=re.sub(r'\bx ontrol\b', 'control', s)
    s=re.sub(r'\boutler s\b', 'outlets', s)
    s=re.sub(r'\bpeep hole', 'peephole', s)
    s=re.sub(r'\bwater pik\b', 'waterpik', s)
    s=re.sub(r'\bwaterpi k\b', 'waterpik', s)
    s=re.sub(r'\bplex[iy] glass\b', 'plexiglass', s)
    s=re.sub(r'\bsheet rock\b', 'sheetrock',s)
    s=re.sub(r'\bgen purp\b', 'general purpose',s)
    s=re.sub(r'\bquicker crete\b', 'quikrete',s)
    s=re.sub(r'\bref ridge\b', 'refrigerator',s)
    s=re.sub(r'\bshark bite\b', 'sharkbite',s)
    s=re.sub(r'\buni door\b', 'unidoor',s)
    s=re.sub(r'\bair tit\b','airtight', s)
    s=re.sub(r'\bde walt\b','dewalt', s)
    s=re.sub(r'\bwaterpi k\b','waterpik', s)
    s=re.sub(r'\bsaw za(ll|w)\b','sawzall', s)
    s=re.sub(r'\blg elec\b', 'lg', s)
    s=re.sub(r'\bhumming bird\b', 'hummingbird', s)
    s=re.sub(r'\bde ice(?=r|\b)', 'deice',s)  
    s=re.sub(r'\bliquid nail\b', 'liquid nails', s)  
    s=re.sub(r'\bdeck over\b','deckover', s)
    s=re.sub(r'\bcounter sink(?=s|\b)','countersink', s)
    s=re.sub(r'\bpipes line(?=s|\b)','pipeline', s)
    s=re.sub(r'\bbook case(?=s|\b)','bookcase', s)
    s=re.sub(r'\bwalkie talkie\b','2 pair radio', s)
    s=re.sub(r'(?<=^)ks\b', 'kwikset',s)
    s=re.sub('(?<=[0-9])[\ ]*ft(?=[a-z])', 'ft ', s)
    s=re.sub('(?<=[0-9])[\ ]*mm(?=[a-z])', 'mm ', s)
    s=re.sub('(?<=[0-9])[\ ]*cm(?=[a-z])', 'cm ', s)
    s=re.sub('(?<=[0-9])[\ ]*inch(es)*(?=[a-z])', 'in ', s)
    
    s=re.sub(r'(?<=[1-9]) pac\b', 'pack', s)
 
    s=re.sub(r'\bcfl bulbs\b', 'cfl light bulbs', s)
    s=re.sub(r' cfl(?=$)', ' cfl light bulb', s)
    s=re.sub(r'candelabra cfl 4 pack', 'candelabra cfl light bulb 4 pack', s)
    s=re.sub(r'\bthhn(?=$|\ [0-9]|\ [a-rtuvx-z])', 'thhn wire', s)
    s=re.sub(r'\bplay ground\b', 'playground',s)
    s=re.sub(r'\bemt\b', 'emt electrical metallic tube',s)
    s=re.sub(r'\boutdoor dining se\b', 'outdoor dining set',s)
    
     
    if "a/c" in s:
        if ('unit' in s) or ('frost' in s) or ('duct' in s) or ('filt' in s) or ('vent' in s) or ('clean' in s) or ('vent' in s) or ('portab' in s):
            s=s.replace("a/c","air conditioner")
        else:
            s=s.replace("a/c","ac")

   
    external_data_dict={'airvents': 'air vents', 
    'antivibration': 'anti vibration', 
    'autofeeder': 'auto feeder', 
    'backbrace': 'back brace', 
    'behroil': 'behr oil', 
    'behrwooden': 'behr wooden', 
    'brownswitch': 'brown switch', 
    'byefold': 'bifold', 
    'canapu': 'canopy', 
    'cleanerakline': 'cleaner alkaline',
    'colared': 'colored', 
    'comercialcarpet': 'commercial carpet', 
    'dcon': 'd con', 
    'doorsmoocher': 'door smoocher', 
    'dreme': 'dremel', 
    'ecobulb': 'eco bulb', 
    'fantdoors': 'fan doors', 
    'gallondrywall': 'gallon drywall', 
    'geotextile': 'geo textile', 
    'hallodoor': 'hallo door', 
    'heatgasget': 'heat gasket', 
    'ilumination': 'illumination', 
    'insol': 'insulation', 
    'instock': 'in stock', 
    'joisthangers': 'joist hangers', 
    'kalkey': 'kelkay', 
    'kohlerdrop': 'kohler drop', 
    'kti': 'kit', 
    'laminet': 'laminate', 
    'mandoors': 'main doors', 
    'mountspacesaver': 'mount space saver', 
    'reffridge': 'refrigerator', 
    'refrig': 'refrigerator', 
    'reliabilt': 'reliability', 
    'replaclacemt': 'replacement', 
    'searchgalvanized': 'search galvanized', 
    'seedeater': 'seed eater', 
    'showerstorage': 'shower storage', 
    'straitline': 'straight line', 
    'subpumps': 'sub pumps', 
    'thromastate': 'thermostat', 
    'topsealer': 'top sealer', 
    'underlay': 'underlayment',
    'vdk': 'bdk', 
    'wallprimer': 'wall primer', 
    'weedbgon': 'weed b gon', 
    'weedeaters': 'weed eaters', 
    'weedwacker': 'weed wacker', 
    'wesleyspruce': 'wesley spruce', 
    'worklite': 'work light'}
     
    for word in external_data_dict.keys():
        s=re.sub(r'\b'+word+r'\b',external_data_dict[word], s)
        
    ############ replace words from dict
    for word in automatic_spell_check_dict.keys():
        s=re.sub(r'\b'+word+r'\b',automatic_spell_check_dict[word], s)
           
    return s

##### end of dunction 'spell_correction'
############################################


### another replacement dict used independently
another_replacement_dict={"undercabinet": "under cabinet", 
"snowerblower": "snower blower", 
"mountreading": "mount reading", 
"zeroturn": "zero turn", 
"stemcartridge": "stem cartridge", 
"greecianmarble": "greecian marble", 
"outdoorfurniture": "outdoor furniture", 
"outdoorlounge": "outdoor lounge", 
"heaterconditioner": "heater conditioner", 
"heater/conditioner": "heater conditioner", 
"conditioner/heater": "conditioner heater", 
"airconditioner": "air conditioner", 
"snowbl": "snow bl", 
"plexigla": "plexi gla", 
"whirlpoolga": "whirlpool ga", 
"whirlpoolstainless": "whirlpool stainless", 
"sedgehamm": "sledge hamm", 
"childproof": "child proof", 
"flatbraces": "flat braces", 
"zmax": "z max", 
"gal vanized": "galvanized", 
"battery powere weedeater": "battery power weed eater", 
"shark bite": "sharkbite", 
"rigid saw": "ridgid saw", 
"black decke": "black and decker", 
"exteriorpaint": "exterior paint", 
"fuelpellets": "fuel pellet", 
"cabinetwithouttops": "cabinet without tops", 
"castiron": "cast iron", 
"pfistersaxton": "pfister saxton ", 
"splitbolt": "split bolt", 
"soundfroofing": "sound froofing", 
"cornershower": "corner shower", 
"stronglus": "strong lus", 
"shopvac": "shop vac", 
"shoplight": "shop light", 
"airconditioner": "air conditioner", 
"whirlpoolga": "whirlpool ga", 
"whirlpoolstainless": "whirlpool stainless", 
"snowblower": "snow blower", 
"plexigla": "plexi gla", 
"trashcan": "trash can", 
"mountspacesaver": "mount space saver", 
"undercounter": "under counter", 
"stairtreads": "stair tread", 
"techni soil": "technisoil", 
"in sulated": "insulated", 
"closet maid": "closetmaid", 
"we mo": "wemo", 
"weather tech": "weathertech", 
"weather vane": "weathervane", 
"versa tube": "versatube", 
"versa bond": "versabond", 
"in termatic": "intermatic", 
"therma cell": "thermacell", 
"tuff screen": "tuffscreen", 
"sani flo": "saniflo", 
"timber lok": "timberlok", 
"thresh hold": "threshold", 
"yardguard": "yardgard", 
"incyh": "in.", 
"diswasher": "dishwasher", 
"closetmade": "closetmaid", 
"repir": "repair", 
"handycap": "handicap", 
"toliet": "toilet", 
"conditionar": "conditioner", 
"aircondition": "air conditioner", 
"aircondiioner": "air conditioner", 
"comercialcarpet": "commercial carpet", 
"commercail": "commercial", 
"inyl": "vinyl", 
"vinal": "vinyl", 
"vynal": "vinyl", 
"vynik": "vinyl", 
"skill": "skil", 
"whirpool": "whirlpool", 
"glaciar": "glacier", 
"glacie": "glacier", 
"rheum": "rheem", 
"one+": "1", 
"toll": "tool", 
"ceadar": "cedar", 
"shelv": "shelf", 
"toillet": "toilet", 
"toiet": "toilet", 
"toilest": "toilet", 
"toitet": "toilet", 
"ktoilet": "toilet", 
"tiolet": "toilet", 
"tolet": "toilet", 
"eater": "heater", 
"robi": "ryobi", 
"robyi": "ryobi", 
"roybi": "ryobi", 
"rayobi": "ryobi", 
"riobi": "ryobi", 
"screww": "screw", 
"stailess": "stainless", 
"dor": "door", 
"vaccuum": "vacuum", 
"vacum": "vacuum", 
"vaccum": "vacuum", 
"vinal": "vinyl", 
"vynal": "vinyl", 
"vinli": "vinyl", 
"viyl": "vinyl", 
"vynil": "vinyl", 
"vlave": "valve", 
"vlve": "valve", 
"walll": "wall", 
"steal": "steel", 
"stell": "steel", 
"pcv": "pvc", 
"blub": "bulb", 
"ligt": "light", 
"bateri": "battery", 
"kolher": "kohler", 
"fame": "frame", 
"have": "haven", 
"acccessori": "accessory", 
"accecori": "accessory", 
"accesnt": "accessory", 
"accesor": "accessory", 
"accesori": "accessory", 
"accesorio": "accessory", 
"accessori": "accessory", 
"repac": "replacement", 
"repalc": "replacement", 
"repar": "repair", 
"repir": "repair", 
"replacemet": "replacement", 
"replacemetn": "replacement", 
"replacemtn": "replacement", 
"replaclacemt": "replacement", 
"replament": "replacement", 
"toliet": "toilet", 
"skill": "skil", 
"whirpool": "whirlpool", 
"stailess": "stainless", 
"stainlss": "stainless", 
"stainstess": "stainless", 
"jigsaww": "jig saw", 
"woodwen": "wood", 
"pywood": "plywood", 
"woodebn": "wood", 
"repellant": "repellent", 
"concret": "concrete", 
"windos": "window", 
"wndows": "window", 
"wndow": "window", 
"winow": "window", 
"caamera": "camera", 
"sitch": "switch", 
"doort": "door", 
"coller": "cooler", 
"flasheing": "flashing", 
"wiga": "wigan", 
"bathroon": "bath room", 
"sinl": "sink", 
"melimine": "melamine", 
"inyrtior": "interior", 
"tilw": "tile", 
"wheelbarow": "wheelbarrow", 
"pedistal": "pedestal", 
"submerciable": "submercible", 
"weldn": "weld", 
"contaner": "container", 
"webmo": "wemo", 
"genis": "genesis", 
"waxhers": "washer", 
"softners": "softener", 
"sofn": "softener", 
"connecter": "connector", 
"heather": "heater", 
"heatere": "heater", 
"electic": "electric", 
"quarteround": "quarter round", 
"bprder": "border", 
"pannels": "panel", 
"framelessmirror": "frameless mirror", 
"paneling": "panel", 
"controle": "control", 
"flurescent": "fluorescent", 
"flourescent": "fluorescent", 
"molding": "moulding", 
"lattiace": "lattice", 
"barackets": "bracket", 
"vintemp": "vinotemp", 
"vetical": "vertical", 
"verticle": "vertical", 
"vesel": "vessel", 
"versatiube": "versatube", 
"versabon": "versabond", 
"dampr": "damper", 
"vegtable": "vegetable", 
"plannter": "planter", 
"fictures": "fixture", 
"mirros": "mirror", 
"topped": "top", 
"preventor": "breaker", 
"traiter": "trailer", 
"ureka": "eureka", 
"uplihght": "uplight", 
"upholstry": "upholstery", 
"untique": "antique", 
"unsulation": "insulation", 
"unfinushed": "unfinished", 
"verathane": "varathane", 
"ventenatural": "vent natural", 
"shoer": "shower", 
"floorong": "flooring", 
"tsnkless": "tankless", 
"tresers": "dresers", 
"treate": "treated", 
"transparant": "transparent", 
"transormations": "transformation", 
"mast5er": "master", 
"anity": "vanity", 
"tomostat": "thermostat", 
"thromastate": "thermostat", 
"kphler": "kohler", 
"tji": "tpi", 
"cuter": "cutter", 
"medalions": "medallion", 
"tourches": "torch", 
"tighrner": "tightener", 
"thewall": "the wall", 
"thru": "through", 
"wayy": "way", 
"temping": "tamping", 
"outsde": "outdoor", 
"bulbsu": "bulb", 
"ligh": "light", 
"swivrl": "swivel", 
"switchplate": "switch plate", 
"swiss+tech": "swiss tech", 
"sweenys": "sweeney", 
"susbenders": "suspender", 
"cucbi": "cu", 
"gaqs": "gas", 
"structered": "structured", 
"knops": "knob", 
"adopter": "adapter", 
"patr": "part", 
"storeage": "storage", 
"venner": "veneer", 
"veneerstone": "veneer stone", 
"stm": "stem", 
"steqamers": "steamer", 
"latter": "ladder", 
"steele": "steel", 
"builco": "bilco", 
"panals": "panel", 
"grasa": "grass", 
"unners": "runner", 
"maogani": "maogany", 
"sinl": "sink", 
"grat": "grate", 
"showerheards": "shower head", 
"spunge": "sponge", 
"conroller": "controller", 
"cleanerm": "cleaner", 
"preiumer": "primer", 
"fertillzer": "fertilzer", 
"spectrazide": "spectracide", 
"spaonges": "sponge", 
"stoage": "storage", 
"sower": "shower", 
"solor": "solar", 
"sodering": "solder", 
"powerd": "powered", 
"lmapy": "lamp", 
"naturlas": "natural", 
"sodpstone": "soapstone", 
"punp": "pump", 
"blowerr": "blower", 
"medicn": "medicine", 
"slidein": "slide", 
"sjhelf": "shelf", 
"oard": "board", 
"singel": "single", 
"paintr": "paint", 
"silocoln": "silicon", 
"poinsetia": "poinsettia", 
"sammples": "sample", 
"sidelits": "sidelight", 
"nitch": "niche", 
"pendent": "pendant", 
"shopac": "shop vac", 
"shoipping": "shopping", 
"shelfa": "shelf", 
"cabi": "cabinet", 
"nails18": "nail", 
"dewaqlt": "dewalt", 
"barreir": "barrier", 
"ilumination": "illumination", 
"mortice": "mortise", 
"lumes": "lumen", 
"blakck": "black", 
"exterieur": "exterior", 
"expsnsion": "expansion", 
"air condit$": "air conditioner", 
"double pole type chf breaker": "double pole type ch breaker", 
"mast 5 er": "master", 
"toilet rak": "toilet rack", 
"govenore": "governor", 
"in wide": "in white", 
"shepard hook": "shepherd hook", 
"frost fee": "frost free", 
"kitchen aide": "kitchen aid", 
"saww horse": "saw horse", 
"weather striping": "weatherstripper", 
"'girls": "girl", 
"girl's": "girl", 
"girls'": "girl", 
"girls": "girl", 
"girlz": "girl", 
"boy's": "boy", 
"boys'": "boy", 
"boys": "boy", 
"men's": "man", 
"mens'": "man", 
"mens": "mam", 
"men": "man", 
"women's": "woman", 
"womens'": "woman", 
"womens": "woman", 
"women": "woman", 
"kid's": "kid", 
"kids'": "kid", 
"kids": "kid", 
"children's": "kid", 
"childrens'": "kid", 
"childrens": "kid", 
"children": "kid", 
"child": "kid", 
"bras": "bra", 
"bicycles": "bike", 
"bicycle": "bike", 
"bikes": "bike", 
"refridgerators": "fridge", 
"refrigerator": "fridge", 
"refrigirator": "fridge", 
"freezer": "fridge", 
"memories": "memory", 
"fragance": "perfume", 
"fragrance": "perfume", 
"cologne": "perfume", 
"anime": "animal", 
"assassinss": "assassin", 
"assassin's": "assassin", 
"assassins": "assassin", 
"bedspreads": "bedspread", 
"shoppe": "shop", 
"extenal": "external", 
"knives": "knife", 
"kitty's": "kitty", 
"levi's": "levi", 
"squared": "square", 
"rachel": "rachael", 
"rechargable": "rechargeable", 
"batteries": "battery", 
"seiko's": "seiko", 
"ounce": "oz"
}
