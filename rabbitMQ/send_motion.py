#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='motion')

body_msg = '37.494775&126.959141&rtsp://admin:123456789a@220.72.73.126:554'

channel.basic_publish(exchange='', routing_key='motion', body=body_msg)
print(" [x] Sent 'motion detect!' body: "+body_msg)
connection.close()
