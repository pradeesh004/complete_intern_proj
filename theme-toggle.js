function switchTheme() {
  const html = document.documentElement;
  const knob = document.getElementById("theme-knob");
  html.classList.toggle("dark");
  knob.classList.toggle("translate-x-5");
}