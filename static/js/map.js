/**
 * ParkSense — map.js
 * ──────────────────
 * 1. Инициализация карты Leaflet (один раз)
 * 2. Управление маркерами (показать / скрыть)
 * 3. Polling /status каждые 10 секунд
 * 4. Обновление карточек и попапов без перезагрузки
 */

// ─────────────────────────────────────────────
// Конфигурация парковок
// (координаты дублируются здесь для быстрого
//  центрирования; авторитетный источник — сервер)
// ─────────────────────────────────────────────
const PARKING_META = {
  parking_1: { lat: 51.163706, lon: 71.401294, name: "Кумисбекова 4" },
  parking_2: { lat: 51.13339,  lon: 71.465989, name: "Тәуелсіздік 47А" }
};

// ─────────────────────────────────────────────
// Состояние
// ─────────────────────────────────────────────
let markersVisible = false;          // видны ли маркеры
const markers      = {};             // leaflet-маркеры по id
let   lastData     = {};             // последний /status ответ

// ─────────────────────────────────────────────
// Инициализация карты
// Тайл-слой: CartoDB Dark Matter (тёмная тема)
// ─────────────────────────────────────────────
const map = L.map("map", {
  center: [51.148, 71.435],   // примерный центр между двумя точками
  zoom:   12,
  zoomControl: true
});

L.tileLayer(
  "https://tile2.maps.2gis.com/tiles?x={x}&y={y}&z={z}&v=1&key=58052a4d-75da-4072-81f3-93b01a124fe6",
  {
    attribution: '&copy; <a href="https://2gis.kz">2GIS</a>',
    maxZoom: 19
  }
).addTo(map);


// ─────────────────────────────────────────────
// Утилита: определить цвет по заполненности
// ratio = occupied / total  (0..1)
// ─────────────────────────────────────────────
function getColor(ratio) {
  if (isNaN(ratio) || ratio < 0) return "#5a6070";   // нет данных
  if (ratio >= 0.80) return "#ef4444";               // красный ≥80%
  if (ratio >= 0.50) return "#eab308";               // жёлтый 50–80%
  return "#22c55e";                                  // зелёный <50%
}


// ─────────────────────────────────────────────
// Создать кастомный HTML-маркер
// ─────────────────────────────────────────────
function makeIcon(color, label) {
  return L.divIcon({
    className: "",
    html: `<div class="parking-marker" style="background:${color}">
             <span>${label}</span>
           </div>`,
    iconSize:   [40, 40],
    iconAnchor: [20, 40],
    popupAnchor:[0, -44]
  });
}


// ─────────────────────────────────────────────
// Построить HTML попапа для маркера
// ─────────────────────────────────────────────
function buildPopup(id, data) {
  const total  = data.total  || 0;
  const free   = data.free   || 0;
  const occ    = data.occupied || 0;
  const ratio  = total > 0 ? occ / total : 0;
  const pct    = Math.round(ratio * 100);
  const color  = getColor(ratio);
  const time   = data.last_update ? `Обновлено: ${data.last_update}` : "Нет данных";

  return `
    <div class="popup-title">${data.name}</div>
    <div class="popup-row"><span>Свободно</span>  <strong style="color:#22c55e">${free}</strong></div>
    <div class="popup-row"><span>Занято</span>    <strong style="color:#ef4444">${occ}</strong></div>
    <div class="popup-row"><span>Всего</span>     <strong>${total}</strong></div>
    <div class="popup-row"><span>Заполнено</span> <strong>${pct}%</strong></div>
    <div class="popup-bar-wrap">
      <div class="popup-bar" style="width:${pct}%;background:${color}"></div>
    </div>
    <div style="font-size:11px;color:#5a6070;margin-top:8px">${time}</div>
  `;
}


// ─────────────────────────────────────────────
// Создать маркеры (вызывается один раз при
// первом нажатии кнопки «Показать парковки»)
// ─────────────────────────────────────────────
function createMarkers() {
  for (const [id, meta] of Object.entries(PARKING_META)) {
    const data  = lastData[id] || {};
    const total = (data.free || 0) + (data.occupied || 0);
    const ratio = total > 0 ? (data.occupied || 0) / total : 0;
    const color = getColor(ratio);

    // Буква-подпись: P1 / P2
    const label = id === "parking_1" ? "P1" : "P2";

    const marker = L.marker([meta.lat, meta.lon], { icon: makeIcon(color, label) })
      .addTo(map)
      .bindPopup(buildPopup(id, { ...data, name: meta.name }));

    markers[id] = marker;
  }
}


// ─────────────────────────────────────────────
// Обновить существующие маркеры новыми данными
// ─────────────────────────────────────────────
function refreshMarkers(data) {
  for (const [id, info] of Object.entries(data)) {
    const marker = markers[id];
    if (!marker) continue;

    const meta  = PARKING_META[id];
    const total = info.total || 0;
    const ratio = total > 0 ? info.occupied / total : 0;
    const color = getColor(ratio);
    const label = id === "parking_1" ? "P1" : "P2";

    // Обновить иконку (цвет)
    marker.setIcon(makeIcon(color, label));

    // Обновить попап
    marker.setPopupContent(buildPopup(id, { ...info, name: meta.name }));
  }
}


// ─────────────────────────────────────────────
// Обновить карточки в сайдбаре
// ─────────────────────────────────────────────
function refreshCards(data) {
  for (const [id, info] of Object.entries(data)) {
    const free  = info.free     ?? "—";
    const occ   = info.occupied ?? "—";
    const total = info.total    || 0;
    const ratio = total > 0 ? info.occupied / total : null;
    const color = getColor(ratio ?? -1);
    const pct   = ratio !== null ? Math.round(ratio * 100) : 0;

    // Цифры
    const el = (s) => document.getElementById(s);
    if (el(`free-${id}`))  el(`free-${id}`).textContent  = free;
    if (el(`occ-${id}`))   el(`occ-${id}`).textContent   = occ;
    if (el(`total-${id}`)) el(`total-${id}`).textContent  = total || "—";

    // Цветовой индикатор
    const ind = el(`ind-${id}`);
    if (ind) ind.style.background = color;

    // Полоска заполненности
    const bar = el(`bar-${id}`);
    if (bar) {
      bar.style.width      = `${pct}%`;
      bar.style.background = color;
    }

    // Время обновления
    const timeEl = el(`time-${id}`);
    if (timeEl) {
      timeEl.textContent = info.last_update
        ? `Данные от: ${info.last_update}`
        : "Ожидание данных…";
    }
  }
}


// ─────────────────────────────────────────────
// Polling: GET /status каждые 10 секунд
// ─────────────────────────────────────────────
async function fetchStatus() {
  try {
    const resp = await fetch("/status");
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();

    lastData = data;

    // Обновить карточки всегда
    refreshCards(data);

    // Обновить маркеры только если они видны
    if (markersVisible) {
      refreshMarkers(data);
    }

    // Время последней синхронизации
    const now = new Date();
    document.getElementById("lastSync").textContent =
      `Синхр: ${now.getHours().toString().padStart(2,"0")}:${now.getMinutes().toString().padStart(2,"0")}:${now.getSeconds().toString().padStart(2,"0")}`;

  } catch (e) {
    console.warn("Ошибка получения /status:", e.message);
  }
}

// Первый запрос сразу при загрузке
fetchStatus();

// Повторять каждые 10 секунд
setInterval(fetchStatus, 10_000);


// ─────────────────────────────────────────────
// Кнопка «Показать / Скрыть парковки»
// ─────────────────────────────────────────────
function toggleParkings() {
  markersVisible = !markersVisible;

  const btn     = document.getElementById("toggleBtn");
  const overlay = document.getElementById("mapOverlay");

  if (markersVisible) {
    // Первый показ: создаём маркеры
    if (Object.keys(markers).length === 0) {
      createMarkers();
    } else {
      // Маркеры уже есть — просто добавляем на карту
      for (const m of Object.values(markers)) m.addTo(map);
    }

    // Скрыть оверлей
    overlay.classList.add("hidden");

    // Кнопка — активный стиль
    btn.classList.add("active");
    btn.innerHTML = `<span class="toggle-btn__icon">◉</span> Скрыть парковки`;

    // Подогнать карту под маркеры
    const latlngs = Object.values(PARKING_META).map(m => [m.lat, m.lon]);
    map.fitBounds(latlngs, { padding: [60, 60] });

  } else {
    // Убрать маркеры с карты
    for (const m of Object.values(markers)) map.removeLayer(m);

    // Показать оверлей снова
    overlay.classList.remove("hidden");

    btn.classList.remove("active");
    btn.innerHTML = `<span class="toggle-btn__icon">◎</span> Показать парковки`;
  }
}


// ─────────────────────────────────────────────
// Клик по карточке → центрировать карту и
// открыть попап соответствующего маркера
// ─────────────────────────────────────────────
function focusParking(id) {
  if (!markersVisible) return; // маркеры скрыты — ничего не делаем

  const meta   = PARKING_META[id];
  const marker = markers[id];
  if (!meta || !marker) return;

  map.setView([meta.lat, meta.lon], 15, { animate: true });
  marker.openPopup();
}


// ══════════════════════════════════════════════
// ГРАФИК ИСТОРИИ (Chart.js + GET /history)
// ══════════════════════════════════════════════

let historyChart = null;
let currentHistoryParking = "parking_1";

async function loadHistory(parkingId) {
  try {
    const resp = await fetch(`/history?parking_id=${parkingId}&limit=30`);
    const rows = await resp.json();

    const labels   = rows.map(r => r.recorded_at.slice(11, 16)); // HH:MM
    const freeData = rows.map(r => r.free);
    const occData  = rows.map(r => r.occupied);

    const ctx = document.getElementById("historyChart").getContext("2d");

    if (historyChart) {
      // Обновить существующий график
      historyChart.data.labels            = labels;
      historyChart.data.datasets[0].data  = freeData;
      historyChart.data.datasets[1].data  = occData;
      historyChart.update("none");
    } else {
      // Создать график впервые
      historyChart = new Chart(ctx, {
        type: "line",
        data: {
          labels,
          datasets: [
            {
              label: "Свободно",
              data: freeData,
              borderColor: "#22c55e",
              backgroundColor: "rgba(34,197,94,.1)",
              fill: true,
              tension: 0.4,
              pointRadius: 2,
              borderWidth: 2
            },
            {
              label: "Занято",
              data: occData,
              borderColor: "#ef4444",
              backgroundColor: "rgba(239,68,68,.1)",
              fill: true,
              tension: 0.4,
              pointRadius: 2,
              borderWidth: 2
            }
          ]
        },
        options: {
          responsive: true,
          animation: false,
          plugins: {
            legend: {
              labels: { color: "#5a6070", font: { size: 10 }, boxWidth: 10 }
            }
          },
          scales: {
            x: {
              ticks: { color: "#5a6070", font: { size: 9 }, maxTicksLimit: 6 },
              grid:  { color: "#1f2330" }
            },
            y: {
              ticks: { color: "#5a6070", font: { size: 9 } },
              grid:  { color: "#1f2330" },
              beginAtZero: true
            }
          }
        }
      });
    }
  } catch (e) {
    console.warn("История недоступна:", e.message);
  }
}

// Переключение вкладок графика
function switchHistory(parkingId, btn) {
  currentHistoryParking = parkingId;

  // Переключить активную вкладку
  document.querySelectorAll(".htab").forEach(b => b.classList.remove("htab--active"));
  btn.classList.add("htab--active");

  // Сбросить и перестроить график
  if (historyChart) { historyChart.destroy(); historyChart = null; }
  loadHistory(parkingId);
}

// Загрузить историю при старте
loadHistory(currentHistoryParking);

// Обновлять график вместе с остальными данными
const _origFetch = fetchStatus;
// Добавим обновление графика к polling-циклу
setInterval(() => loadHistory(currentHistoryParking), 30_000); // каждые 30 сек
