#!/usr/bin/env python
import pika

connection = pika.BlockingConnection(pika.URLParameters('amqp://faasoong:tnd@faasoong.iptime.org:5672/'))
    #pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

channel.queue_declare(queue='motion')

channel.basic_publish(exchange='', routing_key='motion', body='37.495323&126.956575&rtsp://admin:123456789a@faasoong.iptime.org:554')
print(" [x] Sent 'motion detect!'")
connection.close()
