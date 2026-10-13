# Recolector de dibujos propios (basado en main.py)
# Guarda cada dibujo en dataset_raw/propios/<clase>/
import os, json, base64, tempfile, glob
from flask import Flask, request, redirect, jsonify

BASE = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE, "dataset_raw", "propios")
with open(os.path.join(BASE, "dataset", "classes.json"), encoding="utf-8") as f:
    CLASES = json.load(f)["classes"]
KEYS = [c["key"] for c in CLASES]
NOMBRES = {c["key"]: f'{c["name"]} {c["symbol"]}' for c in CLASES}

app = Flask(__name__)

def conteo():
    return {k: len(glob.glob(os.path.join(SALIDA, k, "*.png"))) for k in KEYS}

def siguiente_clase():
    # Pide siempre la clase con MENOS dibujos, asi quedan balanceadas
    c = conteo()
    return min(KEYS, key=lambda k: c[k])

HTML = """
<html><head><meta charset="utf-8">
<script src="https://ajax.googleapis.com/ajax/libs/jquery/1.7.1/jquery.min.js"></script>
<script>
  var mousePressed = false, lastX, lastY, ctx;
  function InitThis() {
      ctx = document.getElementById('myCanvas').getContext("2d");
      clearArea();
      $('#myCanvas').mousedown(function (e) { mousePressed = true;
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, false); });
      $('#myCanvas').mousemove(function (e) { if (mousePressed)
          Draw(e.pageX - $(this).offset().left, e.pageY - $(this).offset().top, true); });
      $('#myCanvas').mouseup(function () { mousePressed = false; });
      $('#myCanvas').mouseleave(function () { mousePressed = false; });
  }
  function Draw(x, y, isDown) {
      if (isDown) {
          ctx.beginPath();
          ctx.strokeStyle = 'black';
          ctx.lineWidth = 12;          // mismo grosor que la web de prediccion
          ctx.lineCap = "round";
          ctx.lineJoin = "round";
          ctx.moveTo(lastX, lastY);
          ctx.lineTo(x, y);
          ctx.stroke();
      }
      lastX = x; lastY = y;
  }
  function clearArea() {              // fondo BLANCO (igual que la web de prediccion)
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      ctx.fillStyle = "#ffffff";
      ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
  }
  function prepareImg() {
      document.getElementById('myImage').value = document.getElementById('myCanvas').toDataURL();
  }
</script></head>
<body onload="InitThis();" style="font-family:sans-serif">
  <div align="center">
    <h1>Dibuja: __NOMBRE__</h1>
    <canvas id="myCanvas" width="220" height="220" style="border:2px solid black"></canvas><br><br>
    <button onclick="clearArea();return false;">Borrar</button>
    <form method="post" action="upload" onsubmit="prepareImg();">
      <input name="clase" type="hidden" value="__CLASE__">
      <input id="myImage" name="myImage" type="hidden" value="">
      <input type="submit" value="Enviar">
    </form>
    <p>Dibujos guardados: __TOTAL__</p>
    <table border="1" cellpadding="4">__TABLA__</table>
  </div>
</body></html>
"""

@app.route("/")
def main():
    k = siguiente_clase()
    c = conteo()
    tabla = "".join(f"<tr><td>{NOMBRES[x]}</td><td>{c[x]}</td></tr>" for x in KEYS)
    return (HTML.replace("__NOMBRE__", NOMBRES[k]).replace("__CLASE__", k)
                .replace("__TOTAL__", str(sum(c.values()))).replace("__TABLA__", tabla))

@app.route("/upload", methods=["POST"])
def upload():
    clase = request.form.get("clase")
    img = request.form.get("myImage", "").replace("data:image/png;base64,", "")
    if clase in KEYS and img:
        with tempfile.NamedTemporaryFile(delete=False, mode="w+b", suffix=".png",
                                         dir=os.path.join(SALIDA, clase)) as fh:
            fh.write(base64.b64decode(img))
    return redirect("/", code=302)

@app.route("/conteo")
def ver_conteo():
    return jsonify(conteo())

if __name__ == "__main__":
    for k in KEYS:
        os.makedirs(os.path.join(SALIDA, k), exist_ok=True)
    print("Recolector en http://localhost:5001")
    app.run(port=5001)
