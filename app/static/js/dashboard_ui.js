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
}