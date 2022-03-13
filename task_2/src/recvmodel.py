#coding=utf-8
#参考 C:\Program Files\Python39\Lib\http\server.py

from http.server import BaseHTTPRequestHandler, SimpleHTTPRequestHandler
import cgi
class   PostHandler(SimpleHTTPRequestHandler):
    def do_POST(self):
        print('do post')
        form = cgi.FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD':'POST',
                     'CONTENT_TYPE':self.headers['Content-Type'],
                     }
        )
        print(form)
        self.send_response(200)
        self.end_headers()
        #self.wfile.write(('Client: %s\n ' % str(self.client_address) ).encode())
        #self.wfile.write('User-agent: %s\n' % str(self.headers['user-agent']))
        #self.wfile.write('Path: %s\n'%self.path)
       # self.wfile.write('Form data:\n')
        for field in form.keys():
            field_item = form[field]
            filename = field_item.filename
            filevalue  = field_item.value
            filesize = len(filevalue)#文件大小(字节)
            print( len(filevalue))
            with open('../data/mnist_model.pt','wb') as f:
                f.write(filevalue)
        return
if __name__=='__main__':
    from http.server import HTTPServer
    sever = HTTPServer(('0.0.0.0',8081),PostHandler)
    print( 'Starting server, use <Ctrl-C> to stop')
    sever.serve_forever()