# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from nltk.corpus import stopwords
from nltk import FreqDist
import nltk
from collections import defaultdict
import csv
import re
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "../input"]).decode("utf8"))

#df = pd.read_csv('../input/test.csv')
df = open('../input/test.csv')

def get_word(text):
    word_split = re.compile('^[a-zA-Z]+$')
    return [word.strip().lower() for word in word_split.split(text)]


def clear_stopwords(context):
    letters = re.sub("[^a-zA-Z]", " ", context)
    context = letters.lower().split()
    stopword = set(stopwords.words('english'))
    clear = [c for c in context if c not in stopword]
    return clear

def remove_html(context):
    
    cleaner = re.compile('<.*?>')
    clean_text = re.sub(cleaner,'',context)
    return clean_text

def frequent(context):
    freq = FreqDist(context)
    return freq
    #return sorted(freq,key=lambda x:x[1],reverse=True)

def get_all_tags(df):
    df['tags'].map(lambda x:all_tags.extend(x.split()))
    
meaning_less = ['p','would','could','via','emp','two','must','make',
                'e','c','using','r','vs','versa','based','three']
reader = csv.DictReader(df)
output = open('output.csv','w')
writer=csv.writer(output)
writer.writerow(['id','tags'])
all_file = ['biology.csv','cooking.csv','crypto.csv','diy.csv','robotics.csv']
all_tags = [

'thermocouple', 'key-check-value', 'insect', 'affine-cipher', 'front-door', 
'experimental', 'tls', 'mycology', 'bones', 'humanoid', 'toxicology', 'random-oracle-model',
'tortilla-press', 'windmill', 'quietrock', 'smoothie', 'diy-biology', 'arachnology', 
'senses', 'salad', 'gene-regulation', 'breaker', 'protection', 'meatloaf', 'cabling',
'balkan-cuisine', 'flour-tortilla', 'meatballs', 'smell', 'sequence-assembly', 'jelly',
'meat', 'wireless', 'hs1-siv', 'flavor', 'broccoli', 'pbkdf', 'dissociation-constant', 
'extension-cord', 'phylogenetics', 'rc4', 'speech-processing', 'temporary', 'symbiosis',
'crab', 'hand-tools', 'genetic-diagrams', 'delta', 'hashed-elgamal', 'linguistics', 
'random-number-generator', 'audio', 'durian', 'crawfish', 'i2c', 'confit', 'lab-techniques', 
'blood-pressure', 'fat-metabolism', 'simulator', 'neuroplasticity', 'cocaine', 'ceramic', 
'blood-transfusion', 'root', 'cognition', 'circulatory-system', 'quorum-sensing', 'transition',
'health', 'corned-beef', 'translation', 'real-time', 'neurophysiology', 'research-tools', 
'fatty-acid-synthase', 'plumbing-fixture', 'oven', 'cedar-plank', 'glaze', 'french-press',
'lymph', 'aluminum', 'rotation', 'convergent-encryption', 'sweet-potatoes', 'garage', 
'anonymity', 'french-cuisine', 'brain', 'grounding-and-bonding', 'known-plaintext-attack',
'packaging', 'anecdotal-evidence', 'melamine', 'slides', 'ncbi', 'bread', 'conduit', 'breadcrumbs',
'milk', 'dh-parameters', 'histopathology', 'kohlrabi', 'separating', 'interior', 'torx', 'hanging', 
'indonesian-cuisine', 'hmac', 'cnc', 'blowtorch', 'attack', 'framing', 'wainscoting', 'uncanny-valley',
'junction-box', 'odour', 'quadrature-encoder', 'randomness', 'paprika', 'swiss-roll', 'wash', 'vinegar',
'pier-blocks', 'materials', 'license-key', 'macarons', 'salad-dressing', 'death', 'stand-mixer', 'tcga', 
'lasagna', 'weeds', 'dnapolymerase', 'griddle', 'kitchen-safety', 'walkway', 'vegetable', 'kneading', 
'play-structures', 'path-planning', 'hygiene', 'cilantro', 'roast', 'bench', 'protocol', 'driver', 
'chickpeas', 'induction', 'plastic', 'fondue', 'chai', 'drying', 'pymol', 'veal', 'ramp', 'immunoglobin',
'bioluminescence', 'encoding', 'gnss', 'solar-tubes', 'sex-chromosome', 'homomorphic-encryption', 'robotc', 
'egg-noodles', 'rcservo', 'mayonnaise', 'replay-attack', 'identity-based-encryption', 'chimney',
'turkish-cuisine', 'alkalinity', 'beer', 'hot-tub', 'frying', 'chutney', 'house-wrap', 'test-vectors',
'layout', 'plasticity', 'jicama', 'parsnip', 'replication', 'behavior', 'new-home', 'venting', 'gelatin',
'introns', 'gait', 'hollandaise', 'bolts', 'plaster', 'starch', 'joists', 'human-ear', 'curry', 'carbon-steel',
'zero-knowledge-proofs', 'gardening', 'dietary-restriction', 'srp', 'hardwood', 'timber-framing', 'photoperiod', 
'quartz', 'transdermal', 'ethernet', 'fudge', 'dairy-free', 'food-identification', 'units', 'artificial-intelligence',
'drains', 'granite', 'ardupilot', 'chocolate-truffles', 'ccp4', 'coconut', 'wallpaper', 'ducts', 'kinematics',
'sealing', 'emulsion', 'chacha', 'radiation', 'algebraic-attack', 'floor', 'implantation', 'intelligence',
'dosage-compensation', 'feal', 'genetics', 'yorkshire-puddings', 'science', 'sinkhole', 'sunroom', 'can', 
'kosher-salt', 'clothes-washer', 'md5', 'autoimmune', 'yeast', 'pumpkin', 'shellac', 'grating', 'souffle', 
'quinoa', 'hot-chocolate', 'cheese-making', 'ductless', 'placement', 'amino-acids', 'drywall', 'protocol-analysis',
'publication', 'stairs', 'fitness', 'biological-networks', 'voip', 'genetic-code', 'gauge', 'thickening', 
'ephemeral', 'hair', 'crust', 'bell-peppers', 'sha-3', 'leveling', 'frosting', 'tzatziki', 'forward-secrecy',
'fans', 'diffusion', 'brown-sugar', 'mammals', 'intracellular-transport', 'cutting-boards', 'blinding',
'cooking-myth', 'backer-board', 'beets', 'access-panel', 'biophysics', 'md4', 'flight', 'key-escrow', 'color',
'searing', 'probability', 'conversion', 'pkcs1', 'first-robotics', 'odometry', 'engine', 'pie', 'chips', 
'shacal-2', 'connector', 'permutation', 'stone', 'chronobiology', 'plant-anatomy', 'excavator', 'storm-door', 
'preimage-resistance', 'mistakes', 'prefab', 'drosophila', 'cpvc', 'antigen', 'ants', 'visual-cryptography',
'transportation', 'arduino', 'vegetarianism', 'curtain-rail', 'matlab', 'rangefinder', 'index-of-coincidence',
'biochemistry', 'factoring', 'children', 'bec', 'gwas', 'apples', 'stuffing', 'activerobot', 'backdoors', 'walls', 
'research-design', 'product-recommendation', 'odors', 'brushless-motor', 'catering', 'biomedical-technology', 'toaster',
'butter', 'cfl', 'kefir', 'egress', 'sodium', 'force', 'challenge-response', 'duck', 'ecosystem', 'doors', 
'hypersensitivity', 'error-tolerance', 'screens', 'cream-cheese', 'error-margin', 'software', 'uav', 'avalanche',
'caulking', 'one-time-pad', 'torque', 'coax', 'demography', 'cinderblock', 'diet', 'terminology', 'bio-mechanics', 
'meringue', 'pocket-door', 'rocket', 'finnish-cuisine', 'aes', 'neuroanatomy', 'blum-blum-shub', 'fuse-breaker', 
'protein-engineering', 'post-quantum-cryptography', 'sha-512', 'usb', 'food-preservation', 'testing', 'convection', 
'quadratic-residuosity', 'msg', 'morphology', 'sourdough', 'fiat-shamir', 'concrete-block', 'stepper-motor', 'lidar',
'basement', 'erv', 'overflow', 'general-biology', 'dna-sequencing', 'codon', 'optics', 'baking-powder', 'sprinkler-system',
'aging', 'pairing', 'golgi-body', 'message-recovery', 'cleaning', 'invertebrates', 'solar-thermal', 'gearing', 'knives', 
'malleability', 'hash-tree', 'rhubarb', 'circadian-rhythms', 'speculative', 'control', 'salt', 'drain', 'scallops',
'climate-change', 'melting-chocolate', 'sound-proofing', 'dehydration', '3d-structure', 'planning', 'cement-board', 
'ransomware', 'palaeontology', 'moss', 'ctr', 'acrylic', 'deflection', 'pasta', 'algorithm', 'vitamins', 'playfair', 
'sha-256', 'electrical', 'hardy-weinberg', 'wheat', 'flush-mount', 'data', 'tofu', 'thermometer', 'gluten-free', 'tartare',
'peer-review-journal', 'allele', 'abiogenesis', 'serbian-cuisine', 'squid', 'well', 'uv-exposure', 'air-quality', 
'architecture', 'porch', 'goose', 'mental-poker', 'antibiotic-resistance', 'hardwood-refinishing', 'python', 
'function-evaluation', 'reverse-transcription', 'pond', 'rubber', 'air-leaks', 'bay-leaf', 'otr', 'arithmetic', 
'hermitian-curves', 'tenderizing', 'steak', 'java', 'metabolomics', 'heat-management', 'botulism', 'not-exactly-c',
'african', 'feta', 'chia', 'leaks', 'asn.1', 'pectin', 'neapolitan-pizza', 'fire-hazard', 'genomes', 'plumbing', 'lawn', 
'flour', 'garbage-disposal', 'chicken-wings', 'encryption', 'signal-processing', 'battery', 'population-biology', 'meter',
'small-rnaseq', 'mice', 'ice-maker', 'hvac', 'bijection', 'pet-door', 'gas', 'minestrone', 'pki', 'sheetrock', 'micromouse', 
'pollard-rho', 'speck', 'montgomery-multiplication', 'iris', 'navigation', 'whipper', 'flan', 'laminate-floor', 'spliceosome', 
'hole', 'acoustic-rangefinder', 'lemon', 'water-pressure', 'photography', 'pharmacodynamics', 'half-and-half', 'blower', 'utensils',
'creme-brulee', 'nec', 'taste', 'hardness-assumptions', 'stainless', 'mobile-robot', 'poison', 'gasoline', 'vocabulary', 
'stainless-steel', 'sourdough-starter', 'herbs', 'beef', 'pan', 'calcium', 'human-anatomy', 'gfci', 'scottish-cuisine',
'research', 'professional', 'mexican-cuisine', 'platform', 'siding', 'hotp', 'back-up', 'hominy', 'sociality', 'sponge-cake',
'apple', 'energy', 'paint-removal', 'lightweight', 'caulk', 'aquaculture', 'hard-boiled-eggs', 'cistern', 'database', 
'copper-tubing', 'scientific-literature', 'anatomy', 'ice', 'visual-system', 'caramelization', 'kimchi', 'statistical-test',
'sensation', 'scatology', 'sewer', 'landscaping', 'cardiology', 'sidewalk', 'reduction', 'tub', 'biodiversity', 'retaining-wall',
'osmosis', 'post', 'building', 'password-hashing', 'windows', 'gazebo', 'kombucha', 'current', 'knife-skills', 'thai-cuisine',
'thanksgiving', 'coot', 'dynamixel', 'restaurant', 'transcription', 'tissue-repair', 'self-leveling-concrete', 'dna-helix', 
'olive-oil', 'sar', 'sander', 'blind-baking', 'order-preserving', 'computer-vision', 'skin', 'vex', 'footings', 'brie', 
'floorheating', 'protein-expression', 'hole-repair', 'clothing', 'reliability', 'epidemiology', 'wooden-furniture', 
'hybridization', 'blowfish', 'tiling', 'copy-protection', 'chosen-plaintext-attack', 'tilapia', 'bananas', 'quantum-cryptography',
'porch-swing', 'schnorr-signature', 'restaurant-mimicry', 'access-control', 'argon2', 'botany', 'pizza', 'high-altitude', 'isaac',
'stuck', 'damp-proof-course', 'thickness', 'reverse-genetics', 'infusion', 'complexity', 'hinges', 'sha-3-competition', 'burnt',
'tamales', 'psychoneuropharmacology', 'truecrypt', 'binding-sites', 'cooking-time', 'organs', 'c', 'ham', 'radiator', 'moisture',
'chime', 'asbestos', 'saffron', 'pharmacology', 'human-eye', 'grilling', 'molecular-biology', 'dendrology', 'polish-cuisine', 
'dissection', 'wood-finish', 'structural-biology', 'whiskey', 'stress', 'sip', 'severe-weather', 'tortilla', 'remote-data-checking',
'cocoa', 'french-drain', 'power-backup', 'modular-arithmetic', 'mass-cooking', 'hummus', 'cryonics', 'gene-therapy', 'brick', 
'ladder', 'sociobiology', 'jam', 'exercise', 'proteomics', 'cryptanalysis', 'glider', 'medium', 'sliding-glass-door', 'membrane', 
'surgery', 'twins', 'lymphatic', 'mailbox', 'flambe', 'mrds', 'homework', 'limnology', 'celiac-disease', 'flatbread', 'fortuna', 
'holerodent', 'avocados', 'water-heater', 'horizontal', 'predation', 'wood', 'motion', 'peanut-butter', 'extra-cellular-matrix',
'gutters', 'vfh', 'carbonara', 'uht', 'projector', 'monotone-access-structure', 'bacterial-toxins', 'kebab', 'linoleum', 'idea',
'immunosuppression', 'linux', 'australian-cuisine', 'salmonella', 'bathroom', 'inverse-kinematics', 'mindstorms', 'shrimp', 
'decking', 'kitchen-sink', 'treatment', 'lettuce', 'proofing', 'omelette', 'rc6', 'entropy', 'feistel-network', 'frozen',
'inheritance', 'padding', 'corner-bead', 'camping', 'spicy-hot', 'alkyd', 'dairy', 'cpg', 'barn', 'timed-release', 'lawn-mower',
'temperature', 'evaporative-cooling', 'exhaust-fan', 'statistics', 'influenza', 'brownies', 'winterizing', 'routing', 'crispr',
'nutrition', 'data-wiring', 'tokenization', 'waterproofing', 'sds-page', 'kmac', 'visual-servoing', 'drill', 'gelling-agents', 
'stepper-driver', 'peeling', 'doorbell', 'ductwork', 'hiring', 'varnishing', 'factorization', 'taxonomy', 'congestion', 
'distributed-decryption', 'multi-agent', 'chocolate', 'indian-cuisine', 'opencv', 'putty', 'semantic-security', 'allergies',
'scrambled-eggs', 'pest-control', 'biopython', 'mitochondria', 'eeg', 'dimmer-switch', 'furnace', 'cilia', 'vinyl', 'cell-signaling',
'species', 'cheese', 'vaccination', 'smartcard', 'tongue-and-groove', 'spanish-cuisine', 'garbled-circuits', 'molluscs', 'well-pump',
'kinetics', 'steamed-pudding', 'pole', 'mutations', 'contractors', 'sealer', 'spraypainting', 'bread-pudding', 'ribs', 
'radio-control', 'custard', 'olfaction', 'norx', 'fastener', 'pid', 'enigma', 'thermodynamics', 'dynamics', 'polenta', 'hri',
'information-gain', 'laser', 'encapsulation', 'radon', 'malt', 'phosphorylation', 'structure-prediction', 'breathing', 'hash-signature', 
'cell', 'blanching', 'coding-theory', 'code-compliance', 'wok', 'crc', 'steering', 'roofing', 'migration', 'closet', 'trapdoor',
'soymilk', 'dna-methylation', 'lally', '3d-reconstruction', 'endocrinology', 'truffles', 'diy-vs-pro', 'oregano', 'identification',
'acoustic-cryptanalysis', 'baker-percentage', 'gender', 'flavour-pairings', 'addiction', 'vegetarian', 'rum', 'secure-index', 
'digital-audio', 'circular-saw', 'stoneware', 'paneer', 'main', 'bathroom-fixtures', 'cream', 'dashi', 'mirror', 'microscopy',
'vertebrates', 'allometry', 'cloning', 'truss', 'corrosion', 'washing-machine', 'carbohydrates', 'chili-peppers', 'saddle-valve',
'scales', 'prime-numbers', 'weather-resistant', 'cryptographic-hardware', 'water-filtration', 'synestesia', 'pdb', 'visualization', 
'mascarpone', 'provable-security', 'wet', 'perfect-secrecy', 'xray-crystallography', 'eggplant', 'population-dynamics', 'bulb', 
'chronic', 'tortilla-chips', 'ed25519', 'experiment', 'flooding', 'wood-finishing', 'excreta', 'assay-development', 'haddock', 
'spraying', 'steam', 'snacks', 'range', 'keccak', 'bacon', 'evolution', 'variant', 'sauce', 'smoke-detectors', 'crack', 
'chosen-ciphertext-attack', 'pvc', 'untagged', 'firewood', 'heat', 'commutative-encryption', 'food-safety', 'ecb', 'marinade', 
'celery', 'beans', 'cpu', 'bagels', 'garage-door', 'homeostasis', 'reheating', 'recycling', 'cytoskeleton', 'guava', 'compost',
'gate-post', 'pose', 'compass', 'product-review', 'sanding', 'skirting', 'ecology', 'psychophysics', 'marshmallow', 'grout', 
'polymerase', 'gates', 'sink', 'diabetes-mellitus', 'dreaming', 'valve', 'lipids', 't7-promoter', 'wasabi', 'des', 'frequency-analysis',
'mapping', 'physiology', 'fluorescent', 'global-warming', 'lungs', 'tasting', 'upholstery', 'structure', 'sexuality', 'sushi',
'2nd-preimage-resistance', 'archaea', 'telephone', 'cherries', 'energy-efficiency', 'wifi', 'roux', 'icing', 'drywall-mud',
'rivets', 'shortening', 'cupcakes', 'crocus-sativus', 'pork-chops', 'vestigial', 'roasting', 'sous-vide', 'pwm', 'lcr', 'basics',
'host-proof', 'building-regulations', 'power', 'mousse', 'rrt', 'jerk', 'chlorine', 'dutch-oven', 'chimerism', 'crossover',
'protein-interaction', 'strawberries', 'libsodium', 'gene-synthesis', 'ketchup', 'book-recommendation', 'language',
'regeneration', 'acoustic', 'multi-prime-rsa', 'gravy', 'sensors', 'emergency-preparedness', 'plywood', 'reuse', 'life-history',
'tools', 'pasteurization', 'rust-proofing', 'blinds', 'alarm', 'gnocchi', 's-mime', 'transformation', 'species-distribution', 
'span-tables', 'plant-physiology', 'electric-heat', 'heater', 'ipsec', 'joinery', 'pastry', 'mrna', 'mortar', 'menstrual-cycle', 
'compoung', 'vital-wheat-gluten', 'trench', 'transplantation', 'differential-analysis', 'storage', 'cramer-shoup', 'mdf', 
'copy-number-variation', 'seitan', 'utilities', 'tahini', 'research-process', 'red-blood-cell', 'benzodiazepine', 'education',
'thermostat', 'eax', 'cladistics', 'cost', 'flange', 'pepper', 'ofb', 'genetic-linkage', 'cultural-difference', 'pathophysiology',
'history', 'habitat', 'kerberos', 'interference', 'sepsis', 'spray-foam', 'biofilms', 'algebraic-eraser', 'cinnamon',
'pancetta', 'venison', 'mash', 'sashimi', 'knobs', 'outlets', 'recombinant', 'countertops', 'x25519', 'deoxys', 'sounds', 
'citrus', 'self-assembly-furniture', 'garage-door-opener', 'screws', 'actuator', 'stucco', 'data-association', 'electric', 
'error-propagation', 'books', 'chassis', 'mosquitoes', 'puff-pastry', 'resistor', 'sauerkraut', 'private-set-intersection', 
'chembl', 'stove-top', 'measurement', 'basting', 'drainage', 'brining', 'foil-cooking', 'eggs', 'slope', 'finishing',
'particle-board', 'fridge', 'crudo', 'keyak', 'vegetables', 'clinical-trial', 'meet-in-the-middle-attack', 'sterilisation', 
'immune-system', 'human-evolution', 'veneer', 'reproductive-biology', 'melting', 'functional-encryption', 'rental', 'ethology',
'carport', 'interrupts', 'definition', 'vegetation', 'air-conditioning', '3des', 'cutting', 'neuromodulation', 'synthetic-biology', 
'reproduction', 'slow-cooking', 'epoxy', 'angle-grinder', 'microbiology', 'plums', 'pets', 'grass', 'afci', 'epigenetics',
'frittata', 'automatic', 'chickens', 'central-nervous-system', 'traffic-analysis', 'carbon-monoxide', 'fungus', 'isogeny', 
'tomatoes', 'microcontroller', 'auxology', 'irrigation', 'low-latency', 'soil', 'ravioli', 'bacteriology', 'electrophysiology',
'antipredator-adaptation', 'texture', 'food', 'mars', 'doughnuts', 'drain-waste-vent', 'failure', 'microhood', 'turf', 
'legumes', 'serving', 'extinction', 'clearance', 'parasitism', 'kiwifruit', 'crumble', 'stock', 'prion', 'orientation',
'imaging', 'cmac', 'melon', 'tuning', 'walking-robot', 'indicator', 'water-circulating-heating', 'antique', 'glass-top-range', 
'cajun-cuisine', 'cast-iron', 'installation', 'wpa2-psk', 'chromatin', 'accessibility', 'ciphertext-only-attack', 'data-privacy', 
'electronics', 'labview', 'uk', 'feline', 'trim-carpentry', 'wood-filler', 'budget-cooking', 'spices', 'perception',
'clotted-cream', 'staining', 'backsplash', 'precise-positioning', 'tuna', 'air-filter', 'modes-of-operation', 'information-theory', 
'cooling', 'planting', 'ingredient-selection', 'rsap', 'bioenergetics', 'repainting', 'cartridge', 'adaptation', 'ceviche',
'forensics', 'openssl', 'digestive-system', 'endothelium', 'chinese-cuisine', 'drinks', 'action-potential', 'chestnuts',
'spherification', 'toasting', 'abe', 'bouillon', 'organization', 'deduced-reckoning', 'broiler', 'food-web', 'menu-planning',
'sprouting', 'irobot-create', 'wind', 'kale', 'soldering', 'niederreiter', 'steel', 'exploration', 'dough', 'repair', 'candy',
'classification', 'bechamel', 'kbkdf', 'switch', 'grinder', 'ethnobiology', 'skewers', 'spinach', 'ssh', 'ccm', 'hungarian-cuisine', 
'deterministic-encryption', 'batter', 'dicing', 'spackle', 'landscape-ecology', 'cms', 'cement', 'electroencephalography', 'infection', 
'risotto', 'technique', 'detached-structure', 'buckwheat', 'shortcuts', 'trash-can', 'metal-roof', 'dna-isolation', 'vessel',
'key-derivation', 'ultrasonic-sensors', 'format-preserving', 'water', 'software-obfuscation', 'aluminum-foil', 'gamete', 
'evolutionary-game-theory', 'mitosis', 'water-treatment', 'decryption-oracle', 'entomology', 'oblivious-ram', 'paillier', 
'rijndael', 'creme-anglaise', 'shutters', 'coaxial-cable', 'raspberry-pi', 'pathogenesis', 'evo-devo', 'histology', 'old-house',
'key-generation', 'energy-audit', 'stereo-vision', 'maintenance', 'bandsaw', 'routers', 'table-saw', 'advice', 'line-following', 
'pizza-stone', 'blake2', 'snake-oil-cryptography', 'pate', 'vector-field-histogram', 'sleep', 'cooking-safety', 'retrofit', 
'telomere', 'mac', 'jewish-cuisine', 'dna-replication', 'mold', 'standards', 'dip', 'generator', 'green-fluorescent-protein',
'squash', 'phone-wiring', 'baseboard', 'attic-door', 'cell-sorting', 'beverages', 'halogen', 'flowering', 'heating', 'fire-extinguisher',
'theoretical-biology', 'compression', 'sex', 'pestle', 'human-physiology', 'screwdriver', 'imu', 'cakes', 'biomedical-engineering',
'virology', 'siphash', 'drafting', 'ros', 'distinguisher', 'pseudogenes', 'multiple-encryption', 'allergy', 'stretching-spring',
'mathematical-models', 'side-channel-attack', 'authentication', 'skipjack', 'bundled-cables', 'brussels-sprouts', 'hyperplasia',
'arm', 'television', 'md2', 'apoptosis', 'solar-panels', 'caffeine', 'tweakable-cipher', 'environment', 'egg-whites', 'legged', 
'nut-butters', 'focaccia', 'ntru', 'traitor-tracing', 'duct-tape', 'kitchen-counters', 'steganography', 'please-remove-this-tag', 
'machine-learning', 'decorating', 'tart', 'heart-output', 'angle', 'fence', 'substitution-cipher', 'fan', 'kem', 'protein-structure', 
'frying-pan', 'chorizo', 'homocysteine', 'freezing', 'pathology', 'pain', 'cracks', 'searchable-encryption', 'timing-attack', 
'ceramic-tile', 'electrical-distribution', 'chirality', 'middle-eastern-cuisine', 'asparagus', 'linear-cryptanalysis', 'noncoding-rna',
'prime-field', 'gap', 'deep-dish-pizza', 'acid', 'abs', 'block-cipher', 'carter-wegman', 'laundry', 'rolling', 'tv-antenna', 
'autophagy', 'maillard', 'key-reuse', 'transfer-switch', 'radiant-heating', 'morphometry', 'honeycomb', 'camellia', 'pita', 'trim',
'snp', 'elgamal-encryption', 'american-cuisine', 'vibration', 'dishwasher', 'grounding', 'cable-management', 'uv', 'popcorn',
'wheel', 'replacement', 'oogenesis', 'event-related-potential', 'lennox', 'cell-membrane', 'marble', 'lobster', 'sensor-error',
'passwords', 'embedded-systems', 'rust-removal', 'scratches', 'butchering', 'porridge', 'bile', 'shallots', 'alcohol', 'roomba', 
'pulses', 'hood', 'offal', 'toffee', 'evaporated-milk', 'cabinets', 'teeth', 'medicinal-chemistry', 'universal-re-encryption', 
'walk', 'pop-rocks', 'circuit-breaker', 'pancakes', 'copper-cookware', 'anaerobic-respiration', 'rolls', 'salsa', 'structural',
'analgesia', 'sexual-selection', 'seasoning-pans', 'fish', 'biclique-attack', 'caramel', 'pairings', 'flashing', 'register', 
'alternating-step', 'electric-stoves', 'xsalsa20', 'air-conditioner', 'sex-ratio', 'stir-fry', 'scrypt', 'smoke-flavor', 'mass-spec',
'threefish', '3d-model', 'condensation', 'performance', 'cornflake', 'threshold-cryptography', 'led', 'pine', 'rodents', 'filtering', 
'municipality', 'kosher', 'dry-aging', 'blender', 'low-voltage', 'dumplings', 'nxt', 'conservation-biology', 'constants', 'pilot-light',
'tree', 'salsa20', 'ant', 'condiments', 'transcription-factor', 'glucose-syrup', 'fireplace', 'kalman-filter', 'chromosome', 
'file-format', 'bulk-cooking', 'gynecology', 'schnorr-identification', 'cables', 'xof', 'ligation', 'pen-and-paper', 'force-sensor', 
'cell-biology', 'shingles', 'accelerometer', 'linear-bearing', 'human-genome', 'carrots', 'fastening', 'dopamine', 'spoilage',
'rc2', 'arx', 'stem-cells', 'histone-modifications', 'membrane-transport', 'light', 'dynorphin', 'mate', 'hot-sauce', 'thawing',
'zucchini', 'operons', 'solder', 'antibiotics', 'p1363', 'enzymes', 'braising', 'squeak', 'particle-filter', 'syrup', 'dual-ec-drbg',
'gammon', 'eukaryotic-cells', 'soccer', 'tankless', 'lumber', 'wiring', 'wine', 'certificateless-crypto', 'sunchokes', 
'algorithm-design', 'dead-reckoning', 'microarray', 'crockpot', 'rope-knots', 'cookbook', 'hexapod', 'blocksize', 'jamb',
'cookware', 'vacuum', 'rye', 'predicate-encryption', 'joint', 'skein', 'gcm', 'cbc-mac', 'glue', 'coa', 'electrical-panel',
'pot', 'locks', 'kinect', 'cameras', 'sticky-rice', 'antihistamines', 'mass-spectrometry', 'broth', 'fst', 'saw', 'corn', 
'immunology', 'vigenere', 'block-and-beam', 'charcuterie', 'phosphate', 'errors', 'cholesterol', 'barley', 'unlinkability', 
'sifting', 'ice-cream', 'construction', 'salami', 'patio', 'cbc', 'sandwich', 'discrete-logarithm', 'junction', 'masa',
'dynamic-programming', 'coriander', 'jacobian', 'gps', 'knife-safety', 'puree', 'sex-determination', 'electric-motor', 
'senescence', 'snail', 'pork-shoulder', 'foam', 'multiparty-computation', 'shed', 'pest', 'auv', 'developmental-biology',
'one-time-password', 'crepe', 'underwater', 'handrail', 'animal-models', 'mussels', 'phenology', 'copper', 'flowers',
'propane-grill', 'alcohol-content', 'electrical-stimulation', 'whipped-cream', 'organic', 'earthquake', 'fips-140',
'flow-cytometry', 'appliances', 'sequence-alignment', 'muffins', 'differentiation', 'sauteing', 'grapes', 'remove', 'storage-method',
'spout', 'restriction-enzymes', 'lymphoma', 'ecoli', 'fire-sprinkler', 'number-theory', 'mri', 'ontology', 'bioinorganic-chemistry', 
'breaker-box', 'minerals', 'growth-media', 'network', 'macroevolution', 'vietnamese-cuisine', 'exhaust', 'ugv',
'patching-drywall', 'blood-brain-barrier', 'behaviour', 'german-cuisine', 'cranial-nerves', 'yogurt', 'convenience-foods', 
'astrobiology', 'neutral', 'comparisons', 'virus', 'thermostat-c-wire', 'species-identification', 'quiche', 'zoology', 
'presentation', 'dehumidifier', 'melting-sugar', 'one-way-function', 'onion-routing', 'studs', 'garlic', 'herpetology', 
'adhesive', 'tumeric', 'white-box', 'mhc', 'definitions', 'saliva', 'sstp', 'histamine', 'moulding', 'microwave-oven', 
'healing', 'chain', 'quantitative-genetics', 'homosexuality', 'product-of-exponentials', 'motion-sensor', 'kettle',
'plant-perception', 'hibiscus-tea', 'fondant', 'weep-hole', 'mixnets', 'gene-annotation', 'onions', 'elderberries',
'prokaryotes', 'hallucinogens', 'nutrient-composition', 'ecophysiology', 'aluminum-cookware', 'reference-request', 'dogs',
'seeds', 'loft', 'serpent', '120-240v', 'horizontal-gene-transfer', 'miter', 'embedded', 'barbecue-sauce', 'driveway',
'ring-main', 'nonce', 'greek-cuisine', 'psychology', 'plasmids', 'speakers', 'forest', 'integrity', 'water-damage', 
'recombination', 'transport-security', 'allelopathy', 'stability', 'fryer', 'crawlspace', 'proteins', 'design', 'brain-stem', 
'reflexes', 'grains', '3d-printing', 'fruit-leather', 'crl', 'dpa', 'nose', 'pathway', 'measuring-scales', 'breakfast', 
'propulsion', 'engineered-flooring', 'key-recovery', 'community-ecology', 'skylight', 'dsa', 'planning-permission', 'syskit', 
'classical-cipher', 'sweeteners', 'chainsaw', 'wasps', 'basement-refinishing', 'cryptdb', 'neural-engineering', 'goat',
'javascript', 'siv', 'bioinformatics', 'lime', 'pkcs11', 'life', 'sealant', 'rainbow-table', 'disconnect', 'lentivirus',
'population-genetics', 'methods', 'food-science', 'dessert', 'coverage', 'hardwood-floor', 'smoking', 'almond-milk', 
'decomposition', 'yurt', 'hiv', 'respiration', 'purification', 'painting', 'multi-rotor', 'osmoregulation', 'seasonal',
'forgery', 'lamp', 'philosophy-of-science', 'asphalt', 'mint', 'keys', 'chromatography', 'pseudo-random-generator', 
'exterior', 'column', '240v', 'rtfm', 'eyes', 'ecies', 'industrial-robot', 'key-exchange', 'localization', 'seasoning',
'tv', 'radiant-barrier', 'augpake', 'organelle', 'finish', 'spray-paint', 'bedroom', 'proxy-re-encryption', 'melatonin', 
'file-encryption', 'traditional', 'gravel', 'exhaust-vent', 'artichokes', 'oats', 'compressor', 'korean-cuisine',
'deniable-encryption', 'roof', 'oaep', 'biotechnology', 'peaches', 'lepidoptera', 'snake', 'password-based-encryption', 
'carob', 'hardware', 'hot-water', 'birthday-attack', 'mushroom', 'steaming', 'free-range', 'theory', 'spray', 
'alternative-energy', 'doorknob', 'bitslicing', 'foundation', 'humidity', 'laminate', 'plasma-membrane', 'chemistry',
'blood-circulation', 'gustation', 'oil', 'chili', 'health-and-safety', 'insulin', 'locating', 'transfection', 'merkle-damgaard', 
'render', 'crackers', 'sauna', 'sponge', 'flooring', 'old-work', 'pantry', 'marine-biology', 'containers', 'lentils', 
'beam', 'cytokinesis', 'pseudo-random-permutation', 'rice-cooker', 'tuberculosis', 'deadbolt', 'codon-usage', 'connectors',
'swarm', 'embryology', 'pineapple', 'three-pass-protocol', 'furniture', 'protein-evolution', 'decay', 'inspection', 
'remodeling', 'motor', 'nucleic-acids', 'key-rotation', 'public-key', 'biscuits', 'tamarind', 'minipreps', 'renovation',
'dulce-de-leche', 'gazpacho', 'surface-mount', 'literature', 'hamming', 'dust', 'agriculture', 'lighting', 'altruism', 
'molding', 'railing', 'growth', 'ocb', 'man-in-the-middle', 'timestamping', 'key-size', 'key-schedule', 'watering',
'fire', 'collision-resistance', 'soap-dispenser', 'literature-search', 'joints', 'natural-gas', 'microrna',
'distribution-board', 'peanuts', 'sill', 'forward-kinematics', 'substitutions', 'liqueur', 'yolk', 'dna', 'culinary-uses',
'crock', 'tracks', 'centrifugation', 'neurodegerative-disorders', 'melanin', 'acidity', 'chloroplasts', 'caribbean-cuisine', 
'germination', 'aluminum-wiring', 'saving-money', 'rice', 'margarine', 'raw', 'blood-group', 'autonomic-nervous-system', 
'underlayment', 'ribosome', 'inflammation', 'chemical-communication', 'aids', 'sour-cream', 'cord-and-plug', 'exons', 'bitcoin', 
'transposition-cipher', 'leeks', 'pickling', 'bath', 'workshop', 'measuring', 'aspirin', 'marsupials', 'stove', 'level', 
'troubleshooting', 'rock', 'twofish', 'miyaguchi-preneel', 'malaria', 'balance', 'boiling', 'biosynthesis', 'selection',
'automation', 'sugar', 'synapses', 'hill-cipher', 'competent-cells', 'mutton', 'pressure-canner', 'noise-reduction', 'concrete',
'baking', 'blast', 'stomach', 'review', 'partition', 'vanilla', 'mixing', 'human-biology', 'sorbet', 'ontogeny',
'motion-planning', 'u-prove', 'precautions', 'filter', 'crumb-crust', 'adversarial-model', 'rsa', 'cryptoapi', 'fiberglass', 
'elisa', 'fitting', 'rsa-pss', 'lubrication', 'beginner', 'padding-oracle', 'mouse', 'salmon', 'length-extension', 'proof-of-work',
'metabolism', 'child-safety', 'engineering', 'graham-crackers', 'blake2b', 'dryer', 'cookies', 'dangerous', 'slab',
'cultured-food', 'pseudo-random-function', 'closer', 'pedagogy', 'silicone', 'dose', 'rebar', 'tissue', 'memory', 'metal-cutting',
'ebola', 'simon', 'ventilation', 'laptop', 'cucumbers', 'nsa', 'extension', 'defrosting', 'downspout', 'sexual-dimorphism',
'demolition', 'table', 'taffy', 'patch', 'door-frame', 'barbecue', 'mustard', 'hand-blender', 'shower', 'sirs', 'buttermilk', 
'oranges', 'roboti-arm', 'nails', 'knob-and-tube', 'homology', 'rna-interference', 'rabbit', 'surge-suppression', 'esc', 'subpanel', 
'oil-based-paint', 'touch', 'renal-physiology', 'oblivious-transfer', 'ichthyology', 'vanity', 'mosaic', 'fresh', 'ventricles', 
'hamburgers', 'concentration', 'water-purification', 'molecular-genetics', 'kitchen', 'tempering', 'water-quality', 'vinyl-flooring',
'removal', 'neurology', 'stain', 'lead', 'disk-encryption', 'high-throughput', 'autoreceptor', 'reinforcement-learning', 
'h-bridge', 'calibration', 'glucose', 'juicing', 'signature', 'sugar-free', 'immunity', 'consistency', 'humidifier', 'thermophilia',
'clog', 'anchor', 'sha-2', 'lab-reagents', 'pressure-treated', 'charcoal', 'hard-core-predicate', 'authenticated-encryption',
'pohlig-hellman', 'hearing', 'produce', 'bathtub', 'ratio', 'accumulators', 'heat-pump', 'efficiency', 'manufacturing', 'rice-wine', 
'notation', 'vent', 'primer', 'brisket', 'passover', 'mycoplasma', 'artificial-selection', 'elliptic-curves', 'septic-tanks', 
'taps', 'icgc', 'xts', 'invasive-species', 'nuts', 'gingerbread', 'meiosis', 'veterinary-medicine', 'rabin-cryptosystem', 
'asexual-reproduction', 'servos', 'community', 'rna-sequencing', 'cfb', 'nest', 'ornithology', 'information', 'potatoes', 'jerky',
'extracts', 'prokaryotic-cells', 'timer', 'trees', 'damage', 'secure-storage', 'hybrid', 'outdoor', 'boiler', 'powertools', 
'equipment', 'ramen', 'cell-based', 'jalapeno', 'casserole', 'fret', 'cancer', 'image-processing', 'cell-division', 'food-history', 
'pir', 'biga', 'luby-rackoff', 'underground', 'echolocation', 'battle-bot', 'hypothalamus', 'safety', 'welding', 'pharmacokinetics',
'ground', 'ground-beef', 'termite', 'cheesecake', 'sonar', 'hurricane-panels', 'biostatistics', 'carpentry', 'cell-culture', 'receptor',
'mixing-function', 'present', 'homomorphic-signatures', 'sausages', 'light-fixture', 'probabilistic-encryption', 'geometry',
'diffie-hellman', 'wheeled-robot', 'cabbage', 'addition', 'transfusion', 'air-muscle', 'weatherstripping', 'support', 'pretzels', 
'protein-binding', 'pot-roast', 'locomotion', 'greens', 'circuit', 'biological-control', 'xml-encryption', 'confined-space', 
'cocktails', 'programming-languages', 'shelving', 'carbonation', 'thai', 'ovulation', 'pergola', 'relocating', 'fried-eggs', 
'ghee', 'marinara', 'granola', 'water-tank', 'genomics', 'sieve', 'vapor-barrier', 'custom-cabinetry', 'non-repudiation',
'dormer', 'transformer', 'injury', 'publishing', 'additives', 'ceiling', 'mozzarella', 'leavening', 'gumbo', 'veins',
'russian-cuisine', 'poultry', 'muscles', 'lucas', 'treehouse', 'hall-sensor', 'cubes', 'instinct', 'slam', 'sensory-systems', 
'deep-frying', 'steps', 'shamir-secret-sharing', 'lamb', 'dehydrating', 'solar', 'travertine', 'grading', 'vision', 
'sequence-analysis', 'sexual-reproduction', 'openni', 'insulation', 'measurements', 'cut-of-meat', 'straining', 'propane',
'creme-fraiche', 'fireproof', 'silver', 'c++', 'gastroenterology', 'sat', 'magnetometer', 'flavor-base', 'cork', 'lattice-crypto', 
'washer', 'needham-schroeder', 'hepatitis', 'soup', 'coffee', 'drawers', 'universal-hash', 'peel', 'tile', 'mounting', 'soy',
'biology-misconceptions', 'pot-pie', 'masonry', 'kuka', 'parasitology', 'allium', 'pkcs8', 'seafood', 'differences', 'histone', 
'chopping', 'molecular-evolution', 'cas9', 'pcr', 'ph', 'cold-brew', 'chicken', 'shutoff', 'socket', 'pet-proofing', 
'attic-conversion', 'medicine', 'shades', 'voting', 'nist', 'pulmonology', 'simulation', 'polyploidy', 'forced-air', 
'sha-1', 'serving-suggestion', 'romanian-cuisine', 'poaching', 'plausible-deniability', 'mango', 'hot-dog', 'cake', 
'microwave', 'blind-signature', 'fireblocking', 'marrow', 'dna-repair', 'waffle', 'carpet', 'home-theater', 'hla', 'non-stick',
'lfsr', 'voc', 'stews', 'teratology', 'ekf', 'soda', 'deck', 'low-fat', 'pressure-cooker', 'verifiability', 'brute-force-attack',
'honey', 'metal', 'organic-chemistry', 'winter', 'english-cuisine', 'okra', 'rising', 'matrix-multiplication', 'cdna', 'rain', 
'elgamal-signature', 'reptile', 'dna-damage', 'roast-beef', 'fermentation', 'secret-sharing', 'calories', 'grade', 'mimicry', 
'negligible', 'blockage', 'oxtail', 'digital-rights-management', 'pigmentation', 'liver', 'blueberries', 'xor', 'danfoss', 
'electrocardiography', 'timing', 'toilet', 'cod', 'blocking', 'polyurethane', 'desk', 'olive', 'fascia', 'kangaroo', 'ceiling-fan', 
'pool', 'cactus', 'hkdf', 'hydronic', 'espresso', 'mceliece', 'beagle-bone', 'mechanism', 'shopping', 'memory-hard', 
'chemicals', 'dictionary-attack', 'security-definition', 'home-automation', 'robotic-arm', 'ledger', 'brackets', 'flax',
'almonds', 'blood-sugar', 'artificial-life', 'apple-pie', 'recessed-lighting', 'broadcast-encryption', 'raw-meat', 'waste-disposal',
'symmetric', 'food-processing', 'sump-pump', 'receptacle', 'kitchens', 'pipe', 'opioid', 'manipulator', 'neurotransmitter', 
'rewire', 'shellfish', 'pork-belly', 'cellular', 'differential-privacy', 'chip', 'bits', 'distributed-systems',
'initialization-vector', 'websites', 'kerdi', 'protocol-design', 'cellular-respiration', 'serial', 'gene', 'guide',
'photosynthesis', 'turkey', 'safe-prime', 'logic-control', 'central-heating', 'ransac', 'security', 'septic', 'quadcopter', 
'integration', 'pbkdf-2', 'pork', 'central-vacuum', 'enzyme-kinetics', 'pregnancy', 'milling', 'extermination', 'knapsack',
'attic', 'food-transport', 'prepping', 'resources', 'cost-effective', 'input', 'computational-model', 'protein-folding',
'peripheral-nervous-system', 'lock', 'green', 'tumor', 'wire', 'skillet', 'cloud', 'hematology', 'mavlink', 'human-genetics', 
'chilling', 'cranberries', 'woodworking', 'certificates', 'water-meter', 'backyard', 'noise', 'gyroscope', 'human', 'preparation', 
'congruence', 'snow', 'operating-systems', 'rs232', 'retrovirus', 'implementation', 'finite-field', 'cover', 'ganache',
'universal-composability', 'vodka', 'leak', 'vegan', 'lemonade', 'starter', 'servomotor', 'cornstarch', 'systems-biology', 
'spaghetti', 'grinding', 'carpaccio', 's-boxes', 'sheathing', 'mudding', 'sharpening', 'filling', 'development', 'pomegranate', 
'lemon-juice', 'lifespan', 'refrigerator', 'subfloor', 'paint', 'gel-electrophoresis', 'reprap', 'asian-cuisine',
'frozen-yogurt', 'tea', 's-des', 'hose', 'gene-expression', 'cytogenetics', 'ginger', 'group-theory', 'juice', 'coloring',
'low-carb', 'frost', 'lithium-polymer', 'fruit', 'faucet', 'signalling', 'antibody', 'differential-drive', 'excavation', 
'commitments', 'splicing', 'heart-failure', 'openaccess', 'desx', 'alljoyn', 'load-bearing', 'kerosene', 'avr', 'forward-genetics', 
'learning', 'neuroscience', 'cell-cycle', 'ripemd', 'frame', 'hash', 'ripe', 'curing', 'movement', 'orf', 'nomenclature',
'anthropology', 'brake', 'sardines', 'territoriality', 'poly1305', 'drywall-anchor', 'noodles', 'basil', 'disposal', 'extremophiles', 
'communication', 'trusted-platform-module', 'ultrasound', 'glass', 'built-ins', 'key-wrap', 'pedigree', 'fluorescent-microscopy', 
'italian-cuisine', 'baby-food', 'bcrypt', 'cycle', 'kidney', 'spackling-paste', 'pudding', 'storage-lifetime', 'watermelon', 
'canning', 'soaking', 'woodstove', 'pavers', 'fire-pit', 'microbiome', 'rrna', 'parsley', 'thinset', 'chicken-stock', 'logjam',
'rough-in', 'two-wheeled', 'elliptic-curve-generation', 'digital-cash', 'maple-syrup', 'kit', 'natural-selection', 'serotonin',
'speciation', 'hydration', 'molecular-gastronomy', 'stream-cipher', 'soffit', 'teflon', 'alfredo', 'paella', 'key-distribution',
'electromuscular', 'chip-seq', 'experimental-design', 'bone-biology', 'host-pathogen-interaction', 'animal-husbandry', 
'mushrooms', 'water-softener', 'sensor-fusion', 'parchment', 'draft', 'digestion', 'water-hammer', 'biofeedback', 'baking-soda', 
'raspberries', 'quickbread', 'dissolve', 'rna', 'cauliflower', 'dry-rot', 'pump', 'vinyl-siding', 'isoforms', 'recipe-scaling',
'central-air', 'service', 'histone-deacetylase', 'lath-and-plaster', 'plating', 'taping', 'western-blot', 'collective-behaviour',
'pushfit', 'chicken-breast', 'parmesan', 'pex', 'fats', 'sprinkles', 'faq', 'hearth', 'parquet', 'stain-removal', 'unicity-distance',
'japanese-cuisine', 'mount', 'french-fries', 'biogeography', 'pgp', 'seal', 'outdoor-cooking', 'occupancygrid']
#for file in all_file:
#    df = pd.read_csv('../input/' + file)
#    get_all_tags(df)
#    del df
#print(set(all_tags))
print('Length of all tags : {}'.format(len(all_tags)))
for idx,row in enumerate(reader):
    #title = clear_stopwords(row['title']) ## return list
    word_count = 0
    tfidf = defaultdict(int)
    for word in get_word(row['title']):
        for w in word.split():
            if w in all_tags and w.isalpha():
                word_count +=1
                tfidf[w] +=1
        
    for word in tfidf:
        freq = tfidf[word] / word_count
        tfidf[word] = freq
    pred_title = sorted(tfidf,key=tfidf.get,reverse=True)[:10]
    
    #set_ = list(set(row['content'].split()).intersection(row['title'].split()))
    title = clear_stopwords(row['title'])
    content = remove_html(row['content'])
    #content = clear_stopwords(content)
    
    ## compute tfidf
    c_tfidf = defaultdict(int)
    word_count = 0
    for cont in get_word(content):
        for c in cont.split():
            if c.isalpha() and c in all_tags:
                c_tfidf[c] +=1
                word_count +=1
    for word in c_tfidf:
        freq = c_tfidf[word] / word_count
        c_tfidf[word] = freq
    pred_content = sorted(c_tfidf,key=c_tfidf.get,reverse=True)[:10]
    
    all_pred = {}
    for word in set(pred_title + pred_content):
        all_pred[word] = tfidf.get(word,0) + c_tfidf.get(word,0)
    pred = sorted(all_pred,key=all_pred.get,reverse=True)[:5]
    content = clear_stopwords(content)
    #writer.writerow([row['id'],' '.join(title[:3])])
    common = set(content).intersection(title)
    #common = set(common).intersection(set_)
    if common not in pred:
        list(common).append(pred)
    temp = []
    if len(common) ==0:
        for t in title:
            if t not in meaning_less:
                temp.append(t)
        #print('ID : {} , Title : {}'.format(idx+1,title))
        writer.writerow([row['id'],' '.join(temp)])
    else:
        writer.writerow([row['id'],' '.join(common)])
    #writer.writerow([row['id'],' '.join(set(content).intersection(title))])

