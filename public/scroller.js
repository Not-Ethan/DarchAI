let horizontalScrollContainer = document.querySelector(".horizontal-scroll");
let sections = Array.from(document.querySelectorAll(".scroll-section"));
let scrollInProgress = false;
let onCooldown = false;
function scrollToSection(index) {
  if (scrollInProgress) return;

  scrollInProgress = true;

  onCooldown = false;

  const targetSection = sections[index];
  const scrollDuration = 1000;
  const scrollOptions = {
    left: targetSection.offsetLeft,
    behavior: "smooth",
    duration: scrollDuration
  };
  horizontalScrollContainer.scrollTo(scrollOptions);
  setTimeout(() => {
    scrollInProgress = false;
    onCooldown = true;

    setTimeout(() => {
      onCooldown = false;
    }, 50)

  }, scrollDuration);
}

let currentSection = 0;
document.addEventListener("wheel", (event) => {
  if (scrollInProgress) return;
  if(onCooldown) return;

    if (event.deltaX > 0) {
      currentSection = Math.min(currentSection + 1, sections.length - 1);
    } else if (event.deltaY > 0) {
        currentSection = Math.min(currentSection + 1, sections.length - 1); 
    } else {
      currentSection = Math.max(currentSection - 1, 0);
    }



  
  scrollToSection(currentSection);
});
