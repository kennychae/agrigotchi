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
        refreshCamImage();
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

        refreshCamImage()
    } catch (err) {
        console.log("전체 로그 불러오기 실패:", err);
    }
}

async function loadDiseaseAI() {
  // 최신 결과
  try {
    const res = await fetch("/api/ai/latest");
    const data = await res.json();

    document.getElementById("ai-latest-time-1").innerText = data.timestamp ?? "-";
    document.getElementById("ai-latest-result-1").innerText = renderDiseaseCounts(data.ai_result);

    document.getElementById("ai-latest-time-2").innerText = data.timestamp ?? "-";
    document.getElementById("ai-latest-result-2").innerText = renderDiseaseCounts(data.ai_result2);
  } catch (e) {
    console.log("AI latest 로드 실패:", e);
  }

  // 전체 로그
  try {
    const res = await fetch("/api/ai/all");
    const logs = await res.json();

    const list1 = document.getElementById("ai-log-list-1");
    const list2 = document.getElementById("ai-log-list-2");
    list1.innerHTML = "";
    list2.innerHTML = "";

    for (let i = logs.length - 1; i >= 0; i--) {
      const item = logs[i];

      // 로그1
      const div1 = document.createElement("div");
      div1.className = "ai-log-item";
      div1.innerHTML = `
        <div class="ai-log-time">${item.timestamp ?? "-"}</div>
        <div class="ai-log-result">${renderDiseaseCounts(item.ai_result).replaceAll("\n", "<br>")}</div>
      `;
      list1.appendChild(div1);

      // 로그2
      const div2 = document.createElement("div");
      div2.className = "ai-log-item";
      div2.innerHTML = `
        <div class="ai-log-time">${item.timestamp ?? "-"}</div>
        <div class="ai-log-result">${renderDiseaseCounts(item.ai_result2).replaceAll("\n", "<br>")}</div>
      `;
      list2.appendChild(div2);
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

  const nameMap = {
    "Fresh leaves": "신선한 잎",
    "Fresh leaf": "신선한 잎",
    "Dying leaves": "죽은 잎",
    "Dying leaf": "죽은 잎",
    "OK": "정상",
    "NG": "질병 의심"
  };

  if (obj.result === "OK" || obj.result === "NG") {
    return `${nameMap[obj.result]} (${obj.confidence?.toFixed(2) ?? "-"})`;
  }

  const counts =
    obj.class_counts ||
    obj.classCounts ||
    obj.summary ||
    obj;

  if (!counts || typeof counts !== "object") return "-";

  return Object.entries(counts)
    .map(([k, v]) => `${nameMap[k] ?? k} : ${v}`)
    .join("\n");
}

function refreshCamImage() {
  const img1 = document.getElementById("cam-image-1");
  const img2 = document.getElementById("cam-image-2");
  if (!img1 || !img2) return;

  const t = Date.now(); // 캐시 무력화
  img1.src = `/static/images/leaf_result.jpg?t=${t}`;
  img2.src = `/static/images/fruit_result.jpg?t=${t}`;
}

function refreshLatestImage() {
  const img = document.getElementById("latest-image");
  if (!img) return;

  const t = Date.now();
  img.src = `/static/images/cam1.jpg?t=${t}`;
}

function getCurrentView() {
  const views = ["status", "growth", "disease", "device"];
  for (const v of views) {
    const el = document.getElementById(`content-${v}`);
    if (el && el.style.display !== "none") return v;
  }
  return "status";
}

// 처음 페이지 열 때
refreshLatestImage()
loadAllLogs();

// 주기적으로 갱신 (현재 화면 기준)
setInterval(() => {
  const view = getCurrentView();

  if (view === "disease") {
    refreshCamImage();
    loadDiseaseAI();
  } else if (view === "status") {
    refreshLatestImage()
    loadAllLogs();
  }
}, 10000);