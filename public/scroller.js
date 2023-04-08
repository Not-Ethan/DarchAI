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
  currentSection = index;
  setTimeout(() => {
    scrollInProgress = false;
    onCooldown = true;

    setTimeout(() => {
      onCooldown = false;
    }, 50)

  }, scrollDuration);
  updateDotNavigation();
}

let currentSection = 0;
let scrollTimeout;
let scrollAmount = 0;

function getClosestSection() {
  let closestIndex = 0;
  let minDistance = Infinity;
  
  sections.forEach((section, index) => {
    let distance = Math.abs(horizontalScrollContainer.scrollLeft - section.offsetLeft);
    if(index != currentSection ) distance /= 5;
    if (distance < minDistance) {
      minDistance = distance;
      closestIndex = index;
    }
  });

  return closestIndex;
}

document.addEventListener("wheel", (event) => {
  if (scrollInProgress) return;

  clearTimeout(scrollTimeout);

  scrollAmount += event.deltaY;
  horizontalScrollContainer.scrollLeft += event.deltaY;

  scrollTimeout = setTimeout(() => {
    currentSection = getClosestSection();
    scrollToSection(currentSection);
    scrollAmount = 0;
  }, 100); // You can adjust this value to control how long to wait after the user stops scrolling
});

function updateDotNavigation() {
  const dots = document.querySelectorAll(".dot");
  dots.forEach((dot, index) => {
    if (index === currentSection) {
      dot.classList.add("active");
    } else {
      dot.classList.remove("active");
    }
  });
}

document.querySelectorAll(".dot").forEach((dot) => {
  dot.addEventListener("click", () => {
    if (scrollInProgress) return;
    const index = parseInt(dot.dataset.index, 10);
    currentSection = index;
    scrollToSection(index);
  });
});

updateDotNavigation();

 /* 
function calculateOpacity(scrollPercentage, sectionIndex) {
  // Compute the relative position of the section within the scrollable area
  const sectionPercentage = sectionIndex / (sections.length - 1);

  // Calculate the difference between the current scroll percentage and the section's relative position
  const difference = Math.abs(scrollPercentage - sectionPercentage);

  // Calculate the opacity based on the difference
  return Math.max(1 - difference * 10, 0);
}

horizontalScrollContainer.addEventListener("scroll", () => {
  const scrollPercentage = horizontalScrollContainer.scrollLeft / (horizontalScrollContainer.scrollWidth - horizontalScrollContainer.clientWidth);
  const sections = document.querySelectorAll(".scroll-fade");

  sections.forEach((section, index) => {
    const opacity = calculateOpacity(scrollPercentage, index);

    // Get all inner elements and set their opacity
    Array.from(section.children).forEach((child, index) => {
      child.style.opacity = opacity * (index+1);
    });
  });
});
*/
function calculateOpacityAndScale(scrollPercentage, sectionIndex) {
  // Compute the relative position of the section within the scrollable area
  const sectionPercentage = sectionIndex / (sections.length - 1);

  // Calculate the difference between the current scroll percentage and the section's relative position
  const difference = Math.abs(scrollPercentage - sectionPercentage);

  // Calculate the opacity based on the difference
  const opacity = Math.max(1 - difference * 10, 0);

  // Calculate the scale based on the difference
  const scale = 1 + difference * 3;

  return { opacity, scale };
}

horizontalScrollContainer.addEventListener("scroll", () => {
  const scrollPercentage = horizontalScrollContainer.scrollLeft / (horizontalScrollContainer.scrollWidth - horizontalScrollContainer.clientWidth);
  const sections = document.querySelectorAll(".scroll-zoom");

  sections.forEach((section, index) => {
    const {opacity, scale} = calculateOpacityAndScale(scrollPercentage, index);

    // Get all inner elements and set their opacity
    Array.from(section.children).forEach((child, index) => {
      child.style.opacity = opacity * (index+1);
      child.style.transform = `scale(${scale})`;
    });
  });
});
const sectionColors = [
  "rgb(33, 37, 41)", // color for section 1
  "rgb(33, 37, 41)",
  "rgb(164, 166, 143)",
  "rgb(164, 166, 143)" // color for section 2
  // Add more colors for additional sections
];

horizontalScrollContainer.addEventListener("scroll", () => {
  // Existing code for text animations, etc.

  const scrollPercentage = horizontalScrollContainer.scrollLeft / (horizontalScrollContainer.scrollWidth - horizontalScrollContainer.clientWidth);

  const currentSectionIndex = Math.floor(scrollPercentage * sections.length);
  const nextSectionIndex = Math.min(currentSectionIndex + 1, sections.length - 1);

  const sectionScrollPercentage = (scrollPercentage * sections.length) % 1;

  const currentColor = sectionColors[currentSectionIndex].match(/\d+/g).map(Number);
  const nextColor = sectionColors[nextSectionIndex].match(/\d+/g).map(Number);

  const interpolatedColor = currentColor.map((color, index) => {
    return color + (nextColor[index] - color) * sectionScrollPercentage;
  });

  const bgColor = `rgb(${interpolatedColor[0]}, ${interpolatedColor[1]}, ${interpolatedColor[2]})`;

  sections[currentSectionIndex].style.backgroundColor = bgColor;
});

sectionColors.forEach((color, index) => {
  sections[index].style.backgroundColor = color;
})