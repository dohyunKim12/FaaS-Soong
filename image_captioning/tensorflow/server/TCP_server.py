from model import CNN_Encoder,RNN_Decoder
import tensorflow as tf
import pickle
from socket import *
import struct
import shutil
import time

embedding_dim = 256 #임베딩 차원
units = 512 #노드 개수는 512개
vocab_size = 10000 + 1 #사전크기 50000개 

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()


#체크포인트 불러오기
checkpoint_path = "./checkpoints/final_model1"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  # restoring the latest checkpoint in checkpoint_path
  ckpt.restore(ckpt_manager.latest_checkpoint)

image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)
    
with open('./tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
    
with open('./train_captions.pkl', 'rb') as f:
    train_captions = pickle.load(f)

def calc_max_length(tensor):
    return max(len(t) for t in tensor)

train_seqs = tokenizer.texts_to_sequences(train_captions)
cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')
max_length = calc_max_length(train_seqs)

def evaluate(image):
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result

        dec_input = tf.expand_dims([predicted_id], 0)

    return result

HOST = 'SERVER_IP'
PORT = 'SERVER_PORT'
ADDR = (HOST, PORT)
BUFF_SIZE = 1024
FILE_NAME = './webcam_transfer/clientFile.jpg'
src="/home/seeds/iconms12/tmp/tf_image_captioning/webcam_transfer/clientFile.jpg"
dst="/home/seeds/iconms12/tmp/tf_image_captioning/detect_annomaly/"
while True:
    try:
        serverSocket = socket(AF_INET, SOCK_STREAM)
        serverSocket.bind(ADDR)
        serverSocket.listen(5)

        print("접속대기중")
        clientSocket, addr = serverSocket.accept()
        print("접속완료")    
        count=1
        while True:
            FILE_SIZE = clientSocket.recv(8)
            print(FILE_SIZE)
            #ime.sleep(0.001)
            FILE_SIZE = FILE_SIZE.decode()
            FILE_LEN = 0
            f = open(FILE_NAME, 'wb')
            while True:
                client_file = clientSocket.recv(BUFF_SIZE)
                # print(client_file)

                if not client_file:
                    break

                f.write(client_file)
                FILE_LEN += len(client_file)

                if FILE_LEN == int(FILE_SIZE):
                    break
            f.close()
            print('client : ' + FILE_NAME + ' file transfer')
            image="/home/seeds/iconms12/tmp/tf_image_captioning/webcam_transfer/clientFile.jpg"
            server_msg = evaluate(image) #캡션 전송
            sms_msg=' '.join(server_msg[:-1])
            if 'knife' in server_msg or 'sword' in server_msg or 'scissors' in server_msg:
                shutil.copy2(src,dst+f"detect_knife_{count}.jpg")
                count+=1
                msg=trans(sms_msg)
                send_sms(msg)
            # server_msg = bytes(server_msg, encoding='utf-8')
            # clientSocket.send(server_msg)
            clientSocket.send(sms_msg.encode())
    except Exception as e:
        print(e)
        clientSocket.close()
        serverSocket.close()
        pass