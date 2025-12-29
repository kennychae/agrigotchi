/////////
// setup //
async function completeSetup() {
  const res = await fetch("/setup_complete", { method: "POST" });
  const json = await res.json();

  if (json.status === "ok") {
    alert("설정이 완료되었습니다!");
    window.location.href = "/dashboard";
  } else {
    alert(json.message ?? "설정 완료 실패");
  }
}

async function changePassword() {
  const newPw = document.getElementById("new-password").value;

  if (!newPw) {
    alert("새 비밀번호를 입력해주세요.");
    return;
  }

  const body = new URLSearchParams();
  body.append("new_password", newPw);

  const res = await fetch("/change_password", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  const json = await res.json();
  alert(json.message ?? "비밀번호 변경 완료");
}

async function registerSensor() {
  const name = document.getElementById("device-name").value;
  const ip = document.getElementById("device-ip").value;
  const enabled = document.getElementById("device-enabled").checked;

  const body = new URLSearchParams();
  body.append("device_name", name);
  body.append("device_ip", ip);
  body.append("device_enabled", enabled ? "1" : "0");

  const res = await fetch("/register_sensor", {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: body.toString(),
  });

  const json = await res.json();
  alert(json.message ?? "기기 설정 저장 완료");
}