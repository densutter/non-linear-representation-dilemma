import random
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm



#################################################################################################################################
##                                                                                                                             ##
##  Original Tasks from                                                                                                        ##
##  "Finding Alignments Between Interpretable Causal Variables and Distributed Neural Representations" (Geiger et al., 2024b)  ##
##                                                                                                                             ##
#################################################################################################################################


def randvec(n=50, lower=-0.5, upper=0.5):
    """
    Returns a random vector of length `n`. `w` is ignored.

    """
    return np.array([random.uniform(lower, upper) for i in range(n)])


def create_same_pair(embedding_size):
    vec=randvec(embedding_size)
    return vec,vec


def create_diff_pair(embedding_size):
    similar=True
    while similar:
        vec1=randvec(embedding_size)
        vec2=randvec(embedding_size)
        if not (vec1 == vec2).all():
            similar=False
    return vec1,vec2


def make_dataset_sample_model(embedding_size,variable1=False,variable2=False):
    First_pair_same=random.choice([True,False])
    if First_pair_same:
        First_pair=create_same_pair(embedding_size)
    else:
        First_pair=create_diff_pair(embedding_size)
        
    Second_pair_same=random.choice([True,False])
    if Second_pair_same:
        Second_pair=create_same_pair(embedding_size)
    else:
        Second_pair=create_diff_pair(embedding_size)
    
    modelinput=np.concatenate((First_pair, Second_pair), axis=None)
    if First_pair_same==Second_pair_same:
        label=1.0
    else:
        label=0.0
    return modelinput,label


def make_model_dataset(size,embedding_size,device):
    model_inputs=[]
    labels=[]
    for _ in range(size):
        model_input,label=make_dataset_sample_model(embedding_size)
        model_inputs.append(model_input)
        labels.append(label)
    return torch.tensor(model_inputs, dtype=torch.float32).to(device),torch.tensor(labels, dtype=torch.float32).to(device)


def make_dataset_sample_variable_intervention(embedding_size,variable1=False,variable2=False):
    First_pair_same=random.choice([True,False])
    if First_pair_same:
        First_pair=create_same_pair(embedding_size)
    else:
        First_pair=create_diff_pair(embedding_size)
    source_first_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),(np.zeros(embedding_size),np.zeros(embedding_size))]
    if variable1:
        First_pair_same=random.choice([True,False])
        if First_pair_same:
            source_first_pair=create_same_pair(embedding_size)
        else:
            source_first_pair=create_diff_pair(embedding_size)
        if random.choice([True,False]):
            source_first_pair=[source_first_pair,create_same_pair(embedding_size)]
        else:
            source_first_pair=[source_first_pair,create_diff_pair(embedding_size)]
    
    Second_pair_same=random.choice([True,False])
    if Second_pair_same:
        Second_pair=create_same_pair(embedding_size)
    else:
        Second_pair=create_diff_pair(embedding_size)
    source_second_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),(np.zeros(embedding_size),np.zeros(embedding_size))]
    if variable2:
        Second_pair_same=random.choice([True,False])
        if Second_pair_same:
            source_second_pair=create_same_pair(embedding_size)
        else:
            source_second_pair=create_diff_pair(embedding_size)
        if random.choice([True,False]):
            source_second_pair=[create_same_pair(embedding_size),source_second_pair]
        else:
            source_second_pair=[create_diff_pair(embedding_size),source_second_pair]
    
    modelinput=np.concatenate((First_pair, Second_pair), axis=None)
    source0=np.concatenate(source_first_pair, axis=None)
    source1=np.concatenate(source_second_pair, axis=None)
    if First_pair_same==Second_pair_same:
        label=1.0
    else:
        label=0.0
    return modelinput,label,source0,source1

    
def make_intervention_dataset_variable_intervention_all(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        variable1,variable2=random.choice([(True,False),(False,True),(True,True)])
        base,label,source0,source1=make_dataset_sample_variable_intervention(embedding_size,variable1=variable1,variable2=variable2)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[variable1,variable2]
    return intervention_data


def make_intervention_dataset_variable_intervention_first(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        base,label,source0,source1=make_dataset_sample_variable_intervention(embedding_size,variable1=True,variable2=False)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[True,False]
    return intervention_data


def make_dataset_sample_first_input_intervention(embedding_size):
    First_pair_same=random.choice([True,False])
    if First_pair_same:
        First_pair=create_same_pair(embedding_size)
    else:
        First_pair=create_diff_pair(embedding_size)
    
    Second_pair_same=random.choice([True,False])
    if Second_pair_same:
        Second_pair=create_same_pair(embedding_size)
    else:
        Second_pair=create_diff_pair(embedding_size)

    source=[]
    First_pair_same=random.choice([True,False])
    if First_pair_same: #equal or not in first variable
        if random.choice([True,False]): #Same source pair
            source.append((First_pair[1],First_pair[1]))
        else: #Different source pair
            new_vec=randvec(embedding_size)
            while (new_vec == First_pair[1]).all():
                new_vec=randvec(embedding_size)
            source.append((First_pair[1],new_vec))
            
    else:
        new_vec1=randvec(embedding_size)
        while (new_vec1 == First_pair[1]).all():
            new_vec1=randvec(embedding_size)
        if random.choice([True,False]): #Same source pair
            source.append((new_vec1,new_vec1))
        else: #Different source pair
            new_vec2=randvec(embedding_size)
            while (new_vec2 == new_vec1).all():
                new_vec2=randvec(embedding_size)
            source.append((new_vec1,new_vec2))

    if random.choice([True,False]):
        source.append(create_same_pair(embedding_size))
    else:
        source.append(create_diff_pair(embedding_size))
    
    modelinput=np.concatenate((First_pair, Second_pair), axis=None)
    source=np.concatenate(source, axis=None)
    if First_pair_same==Second_pair_same:
        label=1.0
    else:
        label=0.0
    return modelinput,label,source
    

def make_intervention_dataset_first_input_intervention(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        base,label,source=make_dataset_sample_first_input_intervention(embedding_size)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[True]
    return intervention_data



#################################################
##                                             ##
##  LLM Task (Indirect Object Identification)  ##
##                                             ##
#################################################

    
def make_intervention_dataset_LLM(data,size,tokenizer):
    DAS_data=[]
    for _ in tqdm(range(size)):
        DAS_data.append({})
        base=random.choice(data)
        source=random.choice(data)
        DAS_data[-1]["base"]=tokenizer(base[0])
        if source[1]:
            DAS_data[-1]["label"]=torch.stack([
                tokenizer(base[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(base[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ])
        else:
            DAS_data[-1]["label"]=torch.stack([
                tokenizer(base[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(base[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ])
        DAS_data[-1]["sources"]=[tokenizer(source[0])]
        DAS_data[-1]["intervention"]=[True]
        DAS_data
    return DAS_data


def verify(label_a, label_b, tokenizer):
    tokens_a = tokenizer.tokenize(" "+label_a)
    tokens_b = tokenizer.tokenize(" "+label_b)
    tokens_a_no_space = tokenizer.tokenize(label_a)
    if "".join(tokens_a_no_space) == "".join(tokens_a[1:]):
        return False, tokens_a[0] != tokens_b[0]
    else:
        return True, tokens_a[0] != tokens_b[0]


def Preprocess_IOI_Data(data, tokenizer):
    preprocess=[]

    for ac_data in tqdm(data["ioi_sentences"]):
        label=ac_data.split(" ")[-1]
        cutoff_text=ac_data[:-1*(len(label))-1]
        if "." in ac_data:
            prev_name=ac_data.split(".")[1].split(" ")[1]
        elif "," in ac_data:
            prev_name=ac_data.split(",")[1].split(" ")[1]
        pos_label=ac_data.find(label)
        pos_prev_name=ac_data.find(prev_name)
        if pos_label>pos_prev_name:
            hidden_var=True
            label_true=label
            label_false=prev_name
        else:
            hidden_var=False
            label_true=prev_name
            label_false=label

        label_space, label_diff = verify(label_true, label_false, tokenizer)
        if not label_diff:
            # Both labels are the same, skip
            continue
        if label_space:
            # The tokenizer requires a space in front of the label
            label_true = " " + label_true
            label_false = " " + label_false
        else:
            cutoff_text = cutoff_text + " "
        preprocess.append([cutoff_text,hidden_var,label_true,label_false])
    return preprocess


def Generate_LLM_Eval_Intervention_Data(filename,tokenizer,LLM_test_samples,Intervention_train_size,Intervention_eval_size,Intervention_test_size):
    data=pd.read_parquet(filename, engine='pyarrow')
    if len(data) > 2*Intervention_train_size:
        print(f"Truncating data from {len(data)} to {2*Intervention_train_size}")
        data = data.iloc[:2*Intervention_train_size]
    preprocess=Preprocess_IOI_Data(data,tokenizer)
    
    random.shuffle(preprocess)
    
    split_p1=int(len(preprocess)/10*8)
    split_p2=int(len(preprocess)/10*9)
    preprocessed_train=preprocess[:split_p1]
    preprocessed_eval=preprocess[split_p1:split_p2]
    preprocessed_test=preprocess[split_p2:]
    
    LLM_test_data=[[],[]]
    for dp in tqdm(random.sample(preprocessed_test,LLM_test_samples)):
        LLM_test_data[0].append(tokenizer(dp[0], return_tensors="pt"))
        if dp[1]:
            LLM_test_data[1].append(torch.stack([
                tokenizer(dp[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(dp[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ]))
        else:
            LLM_test_data[1].append(torch.stack([
                tokenizer(dp[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(dp[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ]))
    
    DAS_Train_o=make_intervention_dataset_LLM(preprocessed_train,Intervention_train_size,tokenizer)
    DAS_Eval_o=make_intervention_dataset_LLM(preprocessed_eval,Intervention_eval_size,tokenizer)
    DAS_Test_o=make_intervention_dataset_LLM(preprocessed_test,Intervention_test_size,tokenizer)

    return LLM_test_data,DAS_Train_o,DAS_Eval_o,DAS_Test_o


def Makedataset(baba_templates,abba_templates,Names_Split,Places,Objects,size):
    dataset=[]
    for _ in range(size):
        A=random.choice(Names_Split)
        B=random.choice(Names_Split)
        while A==B:
            B=random.choice(Names_Split)
        PLACE=random.choice(Places)
        OBJECT=random.choice(Objects)
        if random.choice([True,False]):
            text=random.choice(baba_templates)
        else:
            text=random.choice(abba_templates)
        text=text.replace("{A}", A)
        text=text.replace("{B}", B)
        text=text.replace("{OBJECT}", OBJECT)
        text=text.replace("{PLACE}", PLACE)
        dataset.append(text)
    ret={}
    ret["ioi_sentences"]=dataset
    return ret
    

def Generate_LLM_Eval_Intervention_Data_NameSplit(tokenizer,LLM_test_samples,Intervention_train_size,Intervention_eval_size,Intervention_test_size,randomsamples=200000):
    baba_templates = [
        "Then, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {B} and {A} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {B} and {A} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
        "Then, {B} and {A} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
        "Then, {B} and {A} had a long argument, and afterwards {B} said to {A}",
        "After {B} and {A} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
        "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
        "When {B} and {A} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
        "While {B} and {A} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
        "While {B} and {A} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
        "After the lunch, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Afterwards, {B} and {A} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {B} and {A} had a long argument. Afterwards {B} said to {A}",
        "The {PLACE} {B} and {A} went to had a {OBJECT}. {B} gave it to {A}",
        "Friends {B} and {A} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
    ]
    abba_templates = [
        "Then, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {A} and {B} had a lot of fun at the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {A} and {B} were working at the {PLACE}. {B} decided to give a {OBJECT} to {A}",
        "Then, {A} and {B} were thinking about going to the {PLACE}. {B} wanted to give a {OBJECT} to {A}",
        "Then, {A} and {B} had a long argument, and afterwards {B} said to {A}",
        "After {A} and {B} went to the {PLACE}, {B} gave a {OBJECT} to {A}",
        "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give it to {A}",
        "When {A} and {B} got a {OBJECT} at the {PLACE}, {B} decided to give the {OBJECT} to {A}",
        "While {A} and {B} were working at the {PLACE}, {B} gave a {OBJECT} to {A}",
        "While {A} and {B} were commuting to the {PLACE}, {B} gave a {OBJECT} to {A}",
        "After the lunch, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Afterwards, {A} and {B} went to the {PLACE}. {B} gave a {OBJECT} to {A}",
        "Then, {A} and {B} had a long argument. Afterwards {B} said to {A}",
        "The {PLACE} {A} and {B} went to had a {OBJECT}. {B} gave it to {A}",
        "Friends {A} and {B} found a {OBJECT} at the {PLACE}. {B} gave it to {A}",
    ]
    
    Names   = ['Faye', 'Yolanda', 'Fannie', 'Mildred', 'Doris', 'Melissa', 'Shannon', 'Natasha', 'Nellie', 'Stacey', 'Sonia', 'Kelly', 'Marsha', 'Inez', 'Jacqueline', 'Violet', 'Miriam', 'Jerry', 'Brian', 'Stella', 'Rachel', 'Pauline', 'Yvette', 'Rebecca', 'Marilyn', 'Diana', 'Chelsea', 'Kathryn', 'Lula', 'Anne', 'Bernadette', 'Loretta', 'Kevin', 'David', 'Alyssa', 'Gwen', 'Amanda', 'Charlene', 'Irma', 'Megan', 'Stacy', 'Virginia', 'Rosemarie', 'Jessica', 'Paula', 'Richard', 'Jackie', 'Teresa', 'Leona', 'Carol', 'Juana', 'Kristi', 'Evelyn', 'Brandy', 'William', 'Deanna', 'Sarah', 'Eleanor', 'Maggie', 'Harriet', 'Kari', 'Rosa', 'Angie', 'Kristina', 'Melanie', 'Mark', 'Martha', 'Lorraine', 'Agnes', 'Janet', 'Geneva', 'Robin', 'Ella', 'Heidi', 'Suzanne', 'Wilma', 'Velma', 'Pat', 'Estelle', 'Erika', 'Alice', 'Sophia', 'Anita', 'Alicia', 'Carmen', 'Lee', 'Grace', 'Susan', 'Sheila', 'Leigh', 'Mona', 'Kim', 'Jody', 'Mattie', 'Ginger', 'Ethel', 'Theresa', 'Dianne', 'Marian', 'Jessie']
    Places  = ['store', 'theater', 'museum', 'shop', 'beach', 'school', 'mountain', 'desert', 'river', 'cafe', 'library', 'cave', 'forest', 'market', 'zoo', 'park', 'home', 'lake', 'bar', 'gym']
    Objects = ['apple', 'raspberry', 'orange', 'grape', 'blueberry', 'coconut', 'banana', 'plum', 'kiwi', 'papaya', 'cherry', 'pear', 'blackberry', 'pineapple', 'honeydew', 'cantaloupe', 'watermelon', 'mango', 'peach', 'strawberry']
    
    
    lenght_train=int(randomsamples/10*8)
    lenght_eval=int(randomsamples/10*1)
    lenght_test=int(randomsamples/10*1)
    
    split_Names_p1=round(len(Names)/10*6)
    split_Names_p2=round(len(Names)/10*8)
    
    preprocessed_train_names=Names[:split_Names_p1]
    preprocessed_eval_names=Names[split_Names_p1:split_Names_p2]
    preprocessed_test_names=Names[split_Names_p2:]
    
    preprocessed_train=Makedataset(baba_templates,abba_templates,preprocessed_train_names,Places,Objects,lenght_train)
    preprocessed_eval=Makedataset(baba_templates,abba_templates,preprocessed_eval_names,Places,Objects,lenght_eval)
    preprocessed_test=Makedataset(baba_templates,abba_templates,preprocessed_test_names,Places,Objects,lenght_test)
    
    preprocessed_train=Preprocess_IOI_Data(preprocessed_train,tokenizer)
    preprocessed_eval=Preprocess_IOI_Data(preprocessed_eval,tokenizer)
    preprocessed_test=Preprocess_IOI_Data(preprocessed_test,tokenizer)
    
    LLM_test_data=[[],[]]
    for dp in tqdm(random.sample(preprocessed_test,min(LLM_test_samples,len(preprocessed_test)))):
        LLM_test_data[0].append(tokenizer(dp[0], return_tensors="pt"))
        if dp[1]:
            LLM_test_data[1].append(torch.stack([
                tokenizer(dp[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(dp[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ]))
        else:
            LLM_test_data[1].append(torch.stack([
                tokenizer(dp[3], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0],
                tokenizer(dp[2], return_tensors="pt", add_special_tokens=False)["input_ids"][0][0]
            ]))
    
    DAS_Train_o=make_intervention_dataset_LLM(preprocessed_train,Intervention_train_size,tokenizer)
    DAS_Eval_o=make_intervention_dataset_LLM(preprocessed_eval,Intervention_eval_size,tokenizer)
    DAS_Test_o=make_intervention_dataset_LLM(preprocessed_test,Intervention_test_size,tokenizer)

    return LLM_test_data,DAS_Train_o,DAS_Eval_o,DAS_Test_o

###############################
##                           ##
##  AndOrAnd and AndOr Task  ##
##                           ##
###############################


def get_Bool_Result(ABC):
    a=ABC[0]
    b=ABC[1]
    c=ABC[2]
    if (a and b) or (b and c):
        return True
    else:
        return False

        
def make_dataset_sample_model_AndOrAnd(embedding_size):
    output=random.choice([True,False])
    while True:
        First_pair_same=random.choice([True,False])
        if First_pair_same:
            First_pair=create_same_pair(embedding_size)
        else:
            First_pair=create_diff_pair(embedding_size)
            
        Second_pair_same=random.choice([True,False])
        if Second_pair_same:
            Second_pair=create_same_pair(embedding_size)
        else:
            Second_pair=create_diff_pair(embedding_size)
            
        Third_pair_same=random.choice([True,False])
        if Third_pair_same:
            Third_pair=create_same_pair(embedding_size)
        else:
            Third_pair=create_diff_pair(embedding_size)
        
        modelinput=np.concatenate((First_pair, Second_pair,Third_pair), axis=None)
        if get_Bool_Result([First_pair_same, Second_pair_same,Third_pair_same]):
            label=1.0
            if output:
                break
        else:
            label=0.0
            if not output:
                break
                
    return modelinput,label,[First_pair,Second_pair,Third_pair],[First_pair_same,Second_pair_same,Third_pair_same]


def make_model_dataset_AndOrAnd(size,embedding_size,device):
    model_inputs=[]
    labels=[]
    for _ in range(size):
        model_input,label,_,_=make_dataset_sample_model_AndOrAnd(embedding_size)
        model_inputs.append(model_input)
        labels.append(label)
    return torch.tensor(model_inputs, dtype=torch.float32).to(device),torch.tensor(labels, dtype=torch.float32).to(device)


def make_dataset_sample_intervention_AndOrAnd(embedding_size,variable1=False,variable2=False):
    if variable1==False and variable2==False:
        does_change=False
    else:
        does_change=random.choice([True,False])
    modelinput,label,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
    variable1_res=(Sim_Values[0] and Sim_Values[1])
    variable2_res=(Sim_Values[1] and Sim_Values[2])
    while (variable1_res and does_change and not variable1) or (variable2_res and does_change and not variable2):
        modelinput,label,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
        variable1_res=(Sim_Values[0] and Sim_Values[1])
        variable2_res=(Sim_Values[1] and Sim_Values[2])
    base_res=get_Bool_Result(Sim_Values)
    
    if does_change:
        label=1.0-label
    
    source_first_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size))]
    source_second_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size))]
    source1=np.concatenate(source_first_pair, axis=None)
    source2=np.concatenate(source_second_pair, axis=None)
    
    while True:
        variable1_res_new=variable1_res
        variable2_res_new=variable2_res
        if variable1:
            source1,_,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
            variable1_res_new=(Sim_Values[0] and Sim_Values[1])
        if variable2:
            source2,_,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
            variable2_res_new=(Sim_Values[1] and Sim_Values[2])
        if ((variable1_res_new or variable2_res_new)==base_res)and not does_change:
            break
        elif ((variable1_res_new or variable2_res_new)!=base_res)and does_change:
            break
    return modelinput,label,source1,source2


def make_intervention_dataset_AndOrAnd(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        variable1,variable2=random.choice([(True,False),(False,True),(True,True)])
        base,label,source0,source1=make_dataset_sample_intervention_AndOrAnd(embedding_size,variable1=variable1,variable2=variable2)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[variable1,variable2]
    return intervention_data


def make_dataset_sample_intervention_AndOr(embedding_size,variable1=False,variable2=False):
    if variable1==False and variable2==False:
        does_change=False
    else:
        does_change=random.choice([True,False])
    modelinput,label,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
    variable1_res=Sim_Values[1]
    variable2_res=(Sim_Values[0] or Sim_Values[2])
    while ((not variable1_res) and does_change and not variable1) or ((not variable2_res) and does_change and not variable2):
        modelinput,label,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
        variable1_res=Sim_Values[1]
        variable2_res=(Sim_Values[0] or Sim_Values[2])
    base_res=get_Bool_Result(Sim_Values)
    if does_change:
        label=1.0-label

    source_first_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size))]
    source_second_pair=[(np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size)),
                       (np.zeros(embedding_size),np.zeros(embedding_size))]
    source1=np.concatenate(source_first_pair, axis=None)
    source2=np.concatenate(source_second_pair, axis=None)
    
    while True:
        variable1_res_new=variable1_res
        variable2_res_new=variable2_res
        if variable1:
            source1,_,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
            variable1_res_new=Sim_Values[1]
        if variable2:
            source2,_,_,Sim_Values=make_dataset_sample_model_AndOrAnd(embedding_size)
            variable2_res_new=(Sim_Values[0] or Sim_Values[2])
        if ((variable1_res_new and variable2_res_new)==base_res)and not does_change:
            break
        elif ((variable1_res_new and variable2_res_new)!=base_res)and does_change:
            break
    return modelinput,label,source1,source2


def make_intervention_dataset_AndOr(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        variable1,variable2=random.choice([(True,False),(False,True),(True,True)])
        base,label,source0,source1=make_dataset_sample_intervention_AndOr(embedding_size,variable1=variable1,variable2=variable2)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[variable1,variable2]
    return intervention_data



#####################################
##                                 ##
##  AndOrAnd fitted Model trining  ##
##                                 ##
#####################################


def make_intervention_dataset_AndOrAnd_DAS_Fitted(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        variable1,variable2=random.choice([(True,False),(False,True),(True,True),(False,False),(False,False)])
        base,label,source0,source1=make_dataset_sample_intervention_AndOrAnd(embedding_size,variable1=variable1,variable2=variable2)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[variable1,variable2]
    return intervention_data

def make_intervention_dataset_AndOr_DAS_Fitted(size,embedding_size):
    intervention_data=[]
    for _ in range(size):
        variable1,variable2=random.choice([(True,False),(False,True),(True,True),(False,False),(False,False)])
        base,label,source0,source1=make_dataset_sample_intervention_AndOr(embedding_size,variable1=variable1,variable2=variable2)
        intervention_data.append({})
        intervention_data[-1]["base"]=torch.tensor(base, dtype=torch.float32)
        intervention_data[-1]["label"]=torch.tensor(label, dtype=torch.float32)
        intervention_data[-1]["sources"]=torch.tensor([source0,source1], dtype=torch.float32)
        intervention_data[-1]["intervention"]=[variable1,variable2]
    return intervention_data
