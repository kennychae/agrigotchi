// dashboard_ui.js
// 실시간 시계
function updateClock() {
    const now = new Date();
    const timeStr = now.toLocaleString();
    document.getElementById("live-clock").innerText = timeStr;
}
setInterval(updateClock, 1000);
updateClock();

// 화면 전환 기능
function changeView(view) {
    const views = ["status", "growth", "disease", "device"];
    views.forEach(v => {
        const el = document.getElementById(`content-${v}`);
        el.style.display = (v === view) ? "block" : "none";
    });

    // 병충해 탭 들어갈 때 AI 결과 로드
    if (view === "disease") {
        loadDiseaseAI();
    }
}

function addLogItem(data) {
    const logList = document.getElementById("log-list");

    const log = document.createElement("div");
    log.classList.add("log-item");

    log.innerHTML = `
        <div class="log-time">${data.timestamp}</div>
        <div class="log-data">
            온도: ${data.temperature} °C,
            습도: ${data.humidity} %,
            CO₂: ${data.co2} ppm
        </div>
    `;

    logList.appendChild(log);
}

async function loadAllLogs() {
    try {
        const res = await fetch("/api/data/all");
        const logs = await res.json();

        const logList = document.getElementById("log-list");
        logList.innerHTML = ""; // 기존 로그 비우기

        // log 추가
        logs.reverse(); // 오래된 순이므로 역순으로 변경
        for (const item of logs) {
            addLogItem(item);
        }
        refreshCamImage();
    } catch (err) {
        console.log("전체 로그 불러오기 실패:", err);
    }
}

async function loadDiseaseAI() {
  // 최신 결과
  try {
    const res = await fetch("/api/ai/latest");
    const data = await res.json();
    document.getElementById("ai-latest-time").innerText = data.timestamp ?? "-";
    document.getElementById("ai-latest-result").innerText = renderDiseaseCounts(data.ai_result);
  } catch (e) {
    console.log("AI latest 로드 실패:", e);
  }

  // 전체 로그
  try {
    const res = await fetch("/api/ai/all");
    const logs = await res.json();

    const list = document.getElementById("ai-log-list");
    list.innerHTML = "";

    // 최신이 위로 보이게: 뒤에서부터 prepend or 역순 loop
    for (let i = logs.length - 1; i >= 0; i--) {
      const item = logs[i];
      const div = document.createElement("div");
      div.className = "ai-log-item";
      div.innerHTML = `
        <div class="ai-log-time">${item.timestamp ?? "-"}</div>
        <div class="ai-log-result">${renderDiseaseCounts(item.ai_result).replaceAll("\n", "<br>")}</div>
      `;
      list.appendChild(div);
    }
  } catch (e) {
    console.log("AI all 로드 실패:", e);
  }
}

function renderDiseaseCounts(aiResultStr) {
  if (!aiResultStr) return "-";

  let obj;
  try {
    obj = JSON.parse(aiResultStr);
  } catch {
    return aiResultStr; // JSON 아니면 그냥 원문
  }

  const counts = obj.class_counts || obj.classCounts || obj.summary; // 혹시 키가 다를 때 대비
  if (!counts || typeof counts !== "object") return "-";

  const nameMap = {
    "Fresh leaves": "신선한 잎",
    "Fresh leaf": "신선한 잎",
    "Dying leaves": "죽은 잎",
    "Dying leaf": "죽은 잎"
  };

  // "신선한 잎 : 2\n죽은 잎 : 1" 형태로 만들기
  return Object.entries(counts)
    .map(([k, v]) => `${nameMap[k] ?? k} : ${v}`)
    .join("\n");
}

function refreshCamImage() {
    const img = document.getElemnetById("cam-image");
    if (!img) return;
    img.src = "/static/images/cam1.jpg?t=" + Date.now();
}

// 처음 페이지 열 때 전체 로그 불러오기
loadAllLogs();
// 주기적으로 로그 갱신
setInterval(loadALLlogs, 10000);