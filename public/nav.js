document.addEventListener("mousemove", function (event) {
  const navbar = document.querySelector(".navbar");
  const footer = document.querySelector(".footer");
  if (event.clientY <= window.innerHeight * 0.1) {
    navbar.style.opacity = "1";
  } else {
    navbar.style.opacity = "0";
  }

  if(event.clientY >= window.innerHeight * 0.9) {
    footer.style.opacity = "1";
  } else {
    footer.style.opacity = "0";
  }
});
