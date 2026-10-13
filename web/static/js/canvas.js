document.addEventListener("DOMContentLoaded", () => {
  const canvas = document.getElementById("lienzo");
  const ctx = canvas.getContext("2d");
  const grosorSlider = document.getElementById("grosor");
  const btnBorrar = document.getElementById("btnBorrar");
  const btnClasificar = document.getElementById("btnClasificar");
  const inputArchivo = document.getElementById("inputArchivo");
  const resultadoTexto = document.getElementById("resultadoTexto");
  const listaProbabilidades = document.getElementById("listaProbabilidades");

  const seccionFeedback = document.getElementById("seccionFeedback");
  const mensajeFeedback = document.getElementById("mensajeFeedback");
  const botonesFeedback = document.querySelectorAll(".btn-fb");
  const contadorCorrectas = document.getElementById("contadorCorrectas");
  const contadorIncorrectas = document.getElementById("contadorIncorrectas");
  const contadorInvalidas = document.getElementById("contadorInvalidas");
  const tasaAcierto = document.getElementById("tasaAcierto");
  const barraProgreso = document.getElementById("barraFeedbackProgreso");
  const btnReiniciar = document.getElementById("btnReiniciarStats");
  const btnNuevaSesion = document.getElementById("btnNuevaSesion");
  const etiquetaSesion = document.getElementById("etiquetaSesion");
  const gridCapas = document.getElementById("gridCapas");

  // Identificador de espacio y sesion aislado por pestaña o ventana
  const configElem = document.getElementById("appConfig");
  const serverSessionId = configElem ? configElem.dataset.sessionId : ("ses_" + Math.random().toString(36).substring(2, 10));

  let sessionId = sessionStorage.getItem("simbolos_tab_sesion_id");
  if (!sessionId) {
    sessionId = serverSessionId;
    sessionStorage.setItem("simbolos_tab_sesion_id", sessionId);
  }

  if (etiquetaSesion) {
    etiquetaSesion.textContent = sessionId;
  }

  let dibujando = false;
  let ultimoX = 0;
  let ultimoY = 0;
  let ultimaPrediccionId = null;

  function limpiarLienzo() {
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  }

  limpiarLienzo();

  function obtenerPosicion(e) {
    const rect = canvas.getBoundingClientRect();
    const clienteX = e.touches ? e.touches[0].clientX : e.clientX;
    const clienteY = e.touches ? e.touches[0].clientY : e.clientY;
    return {
      x: (clienteX - rect.left) * (canvas.width / rect.width),
      y: (clienteY - rect.top) * (canvas.height / rect.height)
    };
  }

  function iniciarDibujo(e) {
    dibujando = true;
    const pos = obtenerPosicion(e);
    ultimoX = pos.x;
    ultimoY = pos.y;
    ctx.beginPath();
    ctx.fillStyle = "#000000";
    ctx.arc(pos.x, pos.y, parseInt(grosorSlider.value, 10) / 2, 0, Math.PI * 2);
    ctx.fill();
  }

  function dibujar(e) {
    if (!dibujando) return;
    if (e.type.startsWith("touch")) e.preventDefault();

    const pos = obtenerPosicion(e);
    ctx.beginPath();
    ctx.strokeStyle = "#000000";
    ctx.lineWidth = parseInt(grosorSlider.value, 10);
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.moveTo(ultimoX, ultimoY);
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
    ultimoX = pos.x;
    ultimoY = pos.y;
  }

  function terminarDibujo() {
    dibujando = false;
  }

  canvas.addEventListener("mousedown", iniciarDibujo);
  canvas.addEventListener("mousemove", dibujar);
  canvas.addEventListener("mouseup", terminarDibujo);
  canvas.addEventListener("mouseleave", terminarDibujo);

  canvas.addEventListener("touchstart", iniciarDibujo, { passive: false });
  canvas.addEventListener("touchmove", dibujar, { passive: false });
  canvas.addEventListener("touchend", terminarDibujo);

  btnBorrar.addEventListener("click", () => {
    limpiarLienzo();
    resultadoTexto.className = "resultado-vacio";
    resultadoTexto.textContent = "Dibuja un símbolo musical en el lienzo y pulsa clasificar.";
    listaProbabilidades.innerHTML = "";
    seccionFeedback.classList.add("oculta");
    mensajeFeedback.textContent = "";
    ultimaPrediccionId = null;
  });

  function actualizarMetricasFeedback(stats) {
    if (!stats) return;
    contadorCorrectas.textContent = stats.correctas ?? 0;
    contadorIncorrectas.textContent = stats.incorrectas ?? 0;
    contadorInvalidas.textContent = stats.invalidas ?? 0;
    const tasa = stats.tasa_acierto ?? 0;
    tasaAcierto.textContent = `${tasa}%`;
    barraProgreso.style.width = `${tasa}%`;
  }

  async function cargarEstadisticasIniciales() {
    try {
      const resp = await fetch("/feedback", {
        headers: { "X-Session-ID": sessionId }
      });
      if (resp.ok) {
        const stats = await resp.json();
        actualizarMetricasFeedback(stats);
      }
    } catch (err) {
      console.warn("No se pudieron cargar estadísticas previas de la sesión:", err);
    }
  }

  cargarEstadisticasIniciales();

  if (btnReiniciar) {
    btnReiniciar.addEventListener("click", async () => {
      try {
        const resp = await fetch("/feedback/reset", {
          method: "POST",
          headers: { "Content-Type": "application/json", "X-Session-ID": sessionId }
        });
        if (resp.ok) {
          const stats = await resp.json();
          actualizarMetricasFeedback(stats);
          if (mensajeFeedback) mensajeFeedback.textContent = "Contador de sesión reiniciado a cero.";
        }
      } catch (err) {
        console.error("Error al reiniciar contador:", err);
      }
    });
  }

  if (btnNuevaSesion) {
    btnNuevaSesion.addEventListener("click", () => {
      sessionId = "ses_" + Math.random().toString(36).substring(2, 9) + "_" + Date.now().toString(36).substring(4);
      sessionStorage.setItem("simbolos_tab_sesion_id", sessionId);
      if (etiquetaSesion) etiquetaSesion.textContent = sessionId;
      actualizarMetricasFeedback({ correctas: 0, incorrectas: 0, invalidas: 0, total: 0, tasa_acierto: 0.0 });
      if (mensajeFeedback) mensajeFeedback.textContent = "Nuevo espacio de sesión iniciado desde cero.";
    });
  }

  function renderizarCapas(capas) {
    if (!gridCapas) return;
    if (!capas || capas.length === 0) {
      gridCapas.innerHTML = '<div class="capa-vacia">No hay mapas de activación disponibles.</div>';
      return;
    }

    gridCapas.innerHTML = "";
    capas.forEach((c) => {
      const card = document.createElement("div");
      card.className = "tarjeta-capa";
      card.innerHTML = `
        <img src="${c.imagen}" alt="${c.nombre}">
        <span class="titulo-capa">${c.nombre}</span>
        <span class="badge-dimension">${c.dimension}</span>
        <span class="desc-capa">${c.descripcion}</span>
      `;
      gridCapas.appendChild(card);
    });
  }

  async function clasificarImagen(imagenBase64) {
    btnClasificar.disabled = true;
    btnClasificar.textContent = "Clasificando...";
    mensajeFeedback.textContent = "";

    try {
      const respuesta = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Session-ID": sessionId
        },
        body: JSON.stringify({ image: imagenBase64, session_id: sessionId })
      });
      const datos = await respuesta.json();

      if (!datos.success) {
        resultadoTexto.textContent = "Error al clasificar: " + (datos.error || "desconocido");
        return;
      }

      ultimaPrediccionId = datos.prediccion_id;
      const p = datos.prediction;
      resultadoTexto.className = "prediccion-destacada";
      resultadoTexto.textContent = `${p.symbol} ${p.name} (${p.percentage}%)`;

      listaProbabilidades.innerHTML = "";
      datos.top_k.slice(0, 4).forEach((item) => {
        const fila = document.createElement("div");
        fila.className = "fila-prob";
        fila.innerHTML = `<span>${item.symbol} ${item.name}</span><span>${item.percentage}%</span>`;
        listaProbabilidades.appendChild(fila);
      });

      seccionFeedback.classList.remove("oculta");
      botonesFeedback.forEach((b) => (b.disabled = false));

      if (datos.capas_intermedias) {
        renderizarCapas(datos.capas_intermedias);
      }
    } catch (err) {
      console.error(err);
      resultadoTexto.textContent = "Error de conexión con el servidor.";
    } finally {
      btnClasificar.disabled = false;
      btnClasificar.textContent = "Clasificar";
    }
  }

  btnClasificar.addEventListener("click", () => {
    clasificarImagen(canvas.toDataURL("image/png"));
  });

  botonesFeedback.forEach((boton) => {
    boton.addEventListener("click", async () => {
      const tipo = boton.getAttribute("data-tipo");
      if (!tipo) return;

      botonesFeedback.forEach((b) => (b.disabled = true));
      mensajeFeedback.textContent = "Registrando...";

      try {
        const resp = await fetch("/feedback", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "X-Session-ID": sessionId
          },
          body: JSON.stringify({
            tipo: tipo,
            prediccion_id: ultimaPrediccionId,
            session_id: sessionId
          })
        });
        const stats = await resp.json();
        actualizarMetricasFeedback(stats);
        mensajeFeedback.textContent = `Evaluación registrada (${tipo}).`;
      } catch (err) {
        console.error(err);
        mensajeFeedback.textContent = "Error al enviar retroalimentación.";
        botonesFeedback.forEach((b) => (b.disabled = false));
      }
    });
  });

  inputArchivo.addEventListener("change", (e) => {
    const archivo = e.target.files[0];
    if (!archivo) return;

    const lector = new FileReader();
    lector.onload = (evento) => {
      const img = new Image();
      img.onload = () => {
        limpiarLienzo();
        const escala = Math.min((canvas.width * 0.8) / img.width, (canvas.height * 0.8) / img.height);
        const w = img.width * escala;
        const h = img.height * escala;
        const x = (canvas.width - w) / 2;
        const y = (canvas.height - h) / 2;
        ctx.drawImage(img, x, y, w, h);
        clasificarImagen(canvas.toDataURL("image/png"));
      };
      img.src = evento.target.result;
    };
    lector.readAsDataURL(archivo);
  });
});
