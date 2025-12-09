///////////////
// dashboard //
async function loadData() {
    const res = await fetch("/api/data/latest");
    const data = await res.json();

    document.getElementById("temperature").innerText = data.temperature;
    document.getElementById("humidity").innerText = data.humidity;
    document.getElementById("ai_result").innerText = data.ai_result;
}

loadData();

/////////
//setup//
async function completeSetup() {
    const res = await fetch("/setup_complete", { method: "POST" });
    const json = await res.json();
    if (json.status === "ok") {
        alert("설정이 완료되었습니다!");
        window.location.href = "/dashboard";
    }
}

async function changePassword() {
    const new_pw = document.getElementById("new_pw").value;

    const res = await fetch("/change_password", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `new_password=${encodeURIComponent(new_pw)}`
    });

    const json = await res.json();
    alert(json.message);
}

async function registerSensor() {
    const device_id = document.getElementById("device_id").value;

    const res = await fetch("/register_sensor", {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: `device_id=${encodeURIComponent(device_id)}`
    });

    const json = await res.json();
    alert(json.message);
}