from flask import Flask,redirect,session,render_template,request,send_from_directory,jsonify,send_file, make_response;
import os;
from Kmeans_FrancoRuggiero import data;

app = Flask(__name__,template_folder="templates",static_folder="static"); #Nombre de la App y Ubicación de los archivos .HTML
#Key de variable de sesion
secret = "1234";
app.secret_key = secret;

@app.route("/style.css")
def style():
    """Get CSS"""
    return send_from_directory(os.path.join(os.path.join(app.root_path,'static'),'css'),'style.css')
@app.route("/favicon.ico")
def icon():
    """Agrega el favicon"""
    return send_from_directory(os.path.join(app.root_path,'static'),'favicon.ico',mimetype='image/vnd.microsoft.icon')
@app.route("/home")
def home():
    info = data();
    print(info);
    # return info
    return render_template("home.html",info=info);
@app.route("/")
def gotoIndex():
    return redirect("/home"); #Si no especifica nada, va al login

app.run(debug = True,host='localhost',port=8001); #Del 65535 hasta el 1023 están reservados.

# debug = True, El servidor se recarga con cada cambio