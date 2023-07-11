import torch
def check_para(file_name,**kwargs):
    # if file_name is not None:
    #     print(f'output from {file_name}')
    output =''
    for key in kwargs:
        if kwargs[key] is None:
            temp = f'{key} is None\n'
        elif isinstance(kwargs[key],list) or isinstance(kwargs[key],tuple):
            temp=''
            if torch.is_tensor(kwargs[key][0]):
                for idex,i in enumerate(kwargs[key]) :
                    temp_list = f'{key}[{idex}]: {kwargs[key][idex].shape}  '
                    temp +=temp_list
                temp+='\n'
        elif isinstance(kwargs[key],int):
            temp = f'{key}: {kwargs[key]}\n'
        elif isinstance(kwargs[key],dict):
            temp=''
            for key_sub in kwargs[key]:
                if kwargs[key][key_sub] is None:
                    temp_dict = f'{key}[{key_sub}] is None    '
                else:
                    temp_dict = f'{key}[{key_sub}]: {kwargs[key][key_sub].shape}    '
                temp +=temp_dict
            temp+='\n'
        else:
            temp = f'{key}: {kwargs[key].shape}\n'
        output =output+temp
    output = output.strip()
    if file_name is not None:
        output = output + f'       from:{file_name}'
    print(output)

if __name__=='__main__':
    a =b = torch.tensor([1,2,3])
    c = None
    d = [a,b]
    e = 2
    f = {'test':a,'test2':b}
    check_para('my_test',a = a,b = b,c = c,d = d,e = e,f = f)
    # check_para(a = a)
    # check_para(b = b)
    # check_para(c = c)
    # check_para(d = d)