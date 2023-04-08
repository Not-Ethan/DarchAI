document.addEventListener("mousemove", function (event) {
  const navbar = document.querySelector(".navbar");
  const indicator = document.querySelector(".navbar-indicator");
//  const footer = document.querySelector(".footer");
  if (event.clientY <= window.innerHeight * 0.1) {
    navbar.style.opacity = "1";
    indicator.style.opacity = "0";
  } else {
    navbar.style.opacity = "0";
    indicator.style.opacity = "1";
  }
/*
  if(event.clientY >= window.innerHeight * 0.9) {
    footer.style.opacity = "1";
  } else {
    footer.style.opacity = "1";
  }
  */
});
