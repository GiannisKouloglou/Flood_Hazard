from Training import *
from Evaluation import *
def main(param, eval, Muggia, Trieste,  Monfalcone):
    get_wkd = os.getcwd()

    #Results and Model - For Training and Validation
    results_path_dir = get_wkd + "/" + "Results_New"+"/"

    if param=='Training' or param=='training' or param== 'Train' or param== 'train':
        # Delete old Results and continue with training
        if os.path.exists(results_path_dir):
            dir_files_list = os.listdir(results_path_dir)
            for i in range(len(dir_files_list)):
                if dir_files_list[i].split("_")[-1] == 'model':
                    os.remove(results_path_dir+dir_files_list[i])
                    break
        training(Muggia,Trieste, Monfalcone)
        return
    elif param=='Evaluation' or param=='evaluation':
        # Check if Results file exist.
        if os.path.exists(results_path_dir):
            dir_files_list = os.listdir(results_path_dir)
            # Check if model exist
            for i in range(len(dir_files_list)):
                if dir_files_list[i].split("_")[-1] == 'model':
                    model_path = results_path_dir + dir_files_list[i]
                    print("EVALUATION WITH: ",model_path.split("/")[-1])
                    evaluation(model_path, eval, results_path_dir)
                    return
            # If there is no model in Result file, then training
            print("*************NO MODEL IN RESULTS FILE**************")
            print("*************STARTING TRAINING**************")
            training(Muggia, Trieste, Monfalcone)
            # and then do Validation
            print(" Evaluation after Training")
            for i in range(len(dir_files_list)):
                if dir_files_list[i].split("_")[-1] == 'model':
                    model_path = results_path_dir + dir_files_list[i]
                    print("EVALUATION WITH: ",model_path.split("/")[-1])
                    evaluation(model_path,eval, results_path_dir)
                    return
            print("Problem with training!!!")
            return
        else:
            # if there is no Result file, then training
            print("*************NO RESULTS FILE**************")
            print("*************STARTING TRAINING**************")
            training(Muggia, Trieste, Monfalcone)
            print(" Evaluation after Training")
            if os.path.exists(results_path_dir):
                dir_files_list = os.listdir(results_path_dir)
                # Check if model exist
                for i in range(len(dir_files_list)):
                    if dir_files_list[i].split("_")[-1] == 'model':
                        model_path = results_path_dir + dir_files_list[i]
                        print("EVALUATION WITH: ", model_path.split("/")[-1])
                        evaluation(model_path,eval, results_path_dir)
                        return
            else:
                print("Problem with training!!!")
            return

if __name__ == "__main__":
    # pipeline should be Train/train or Evaluation/evaluation
    # pipeline = "train"
    pipeline = "evaluation"
    cwd = os.getcwd()


    muggia = cwd + '/' + 'SatImAn_Data' + '/' + "Muggia-City" + "/"
    trieste = cwd + '/' + 'SatImAn_Data' + '/' + "Trieste" + "/"
    monf = cwd + '/' + 'SatImAn_Data' + '/' + "Monfalcone" + "/"


    Muggia_train = [{'date': '20191115', 'place':'Muggia',"path":muggia},
                    {'date': '20191117', 'place':'Muggia',"path":muggia}
                    ]
    # method for water_depth/mask only for DDPMS, thats why its repeated
    muggia_list = os.listdir(muggia)
    for i in range(len(Muggia_train)):
        matching = [s for s in muggia_list if Muggia_train[i]['date'] in s]
        if len(matching)==0:
            print("There is a need for watermask/depth for this date in Muggia: ",Muggia_train[i]['date'])
            transform_waterbodies_DDPMS(Muggia_train[i])
        else:
            print("There is no need for watermask/depth for this date in Muggia: ",Muggia_train[i]['date'])


    Muggia_evaluation = [{'date': '20181029', 'place':'Muggia',"path":muggia},
                         {'date': '20190924', 'place':'Muggia',"path":muggia},
                         {'date': '20191223', 'place':'Muggia',"path":muggia}
                         ]
    for i in range(len(Muggia_evaluation)):
        matching = [s for s in muggia_list if Muggia_evaluation[i]['date'] in s]
        if len(matching) == 0:
            print("There is a need for watermask/depth for this date in Muggia: ", Muggia_evaluation[i]['date'])
            transform_waterbodies_DDPMS(Muggia_evaluation[i])
        else:
            print("There is no need for watermask/depth for this date in Muggia: ", Muggia_evaluation[i]['date'])


    Trieste_train = [{'date': '20181029', 'place':'Trieste',"path":trieste},
                     {'date': '20191115', 'place':'Trieste',"path":trieste},
                     {'date': '20191223', 'place':'Trieste',"path":trieste}
                     ]
    trieste_list = os.listdir(trieste)
    for i in range(len(Trieste_train)):
        matching = [s for s in trieste_list if Trieste_train[i]['date'] in s]
        if len(matching) == 0:
            print("There is a need for watermask/depth for this date in Trieste: ", Trieste_train[i]['date'])
            transform_waterbodies_DDPMS(Trieste_train[i])
        else:
            print("There is no need for watermask/depth for this date in Trieste: ", Trieste_train[i]['date'])


    Trieste_evaluation = [{'date': '20190923', 'place':'Trieste',"path":trieste},
                          {'date': '20191117', 'place':'Trieste',"path":trieste}
                          ]
    for i in range(len(Trieste_evaluation)):
        matching = [s for s in trieste_list if Trieste_evaluation[i]['date'] in s]
        if len(matching) == 0:
            print("There is a need for watermask/depth for this date in Trieste: ", Trieste_evaluation[i]['date'])
            transform_waterbodies_DDPMS(Trieste_evaluation[i])
        else:
            print("There is no need for watermask/depth for this date in Trieste: ", Trieste_evaluation[i]['date'])

    Monfalcone_train = [ {'date': '20181029', 'place':'Monfalcone',"path":monf},
                         {'date': '20191117', 'place':'Monfalcone',"path":monf},
                         {'date': '20191223', 'place':'Monfalcone',"path":monf}
                         ]
    monfalcone_list = os.listdir(monf)
    for i in range(len(Monfalcone_train)):
        matching = [s for s in monfalcone_list if Monfalcone_train[i]['date'] in s]
        if len(matching) == 0:
            print("There is a need for watermask/depth for this date in Monfalcone: ", Monfalcone_train[i]['date'])
            transform_waterbodies_DDPMS(Monfalcone_train[i])
        else:
            print("There is no need for watermask/depth for this date in Monfalcone: ", Monfalcone_train[i]['date'])

    Monfalcone_evaluation = [{'date': '20190924', 'place':'Monfalcone',"path":monf},
                             {'date': '20191115', 'place':'Monfalcone',"path":monf}]
    for i in range(len(Monfalcone_evaluation)):
        matching = [s for s in monfalcone_list if Monfalcone_evaluation[i]['date'] in s]
        if len(matching) == 0:
            print("There is a need for watermask/depth for this date in Monfalcone: ", Monfalcone_evaluation[i]['date'])
            transform_waterbodies_DDPMS(Monfalcone_evaluation[i])
        else:
            print("There is no need for watermask/depth for this date in Monfalcone: ", Monfalcone_evaluation[i]['date'])

    evaluation_list = [Muggia_evaluation, Trieste_evaluation, Monfalcone_evaluation]
    print("MAIN GENERATED FOR: ",pipeline)
    if pipeline=='Training' or pipeline=='training' or pipeline== 'Train' or pipeline== 'train':
        #second place is for evaluation
        main(pipeline, None, Muggia_train,Trieste_train,Monfalcone_train)
    elif pipeline == 'Evaluation' or pipeline == 'evaluation':
        for z in range(len(evaluation_list)):
            #none are the place, we need all only in training
            main(pipeline, evaluation_list[z], Muggia_train, Trieste_train, Monfalcone_train)


