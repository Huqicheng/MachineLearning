import paddle.v2 as paddle
import csv

# init and pass para
paddle.init(use_gpu=False,trainer_count=1)

# set readers ,(it needs to be further checked,but I can't print the data I got from csv file through reader)
text_reader=paddle.reader.creator.text_file("/home/ztono/Documents/speeds.csv")
buffered_reader=paddle.reader.buffered(text_reader,2345678)

# set dataSet

# get the size of each row in csv file,(but it seems very strange for just print 2 string in each line
# ,and only 517 found)
def getsize():
    csv_open=open("/home/Documents","r")
    csv_reader=csv.reader(csv_open)
    for row in csv_reader:
        return len(row)


csv_speeds_size=getsize()

# config simple module
input_data_layer=paddle.layer.data(name="x",type=paddle.data_type.dense_vector(csv_speeds_size))
output_data_predict=paddle.layer.fc(input=input_data_layer,size=1,act=paddle.activation.Linear())
output_data_layer=paddle.layer.data(name="y",type=paddle.data_type.dense_vector(1))
predict_cost=paddle.layer.square_error_cost(input=output_data_predict,label=output_data_layer)

# save topology
speeds_toloploy=paddle.topology.Topology(layers=output_data_predict)
with open("speeds_toloploy.pkl","wb") as f:
    speeds_toloploy.serialize_for_inference(f)

# set parameters
cost_parameters=paddle.parameters.create(predict_cost)

# set optimizer
speeds_optimizer=paddle.optimizer.Momentum(momentum=0)

# set trainer
speeds_trainer=paddle.trainer.SGD(cost=predict_cost,parameters=cost_parameters,update_equation=speeds_optimizer)

# set map
feeding={'x': 0, 'y': 1}

# start training
speeds_trainer.train(reader=buffered_reader,num_passes=30,event_handler=None,feeding=feeding)
