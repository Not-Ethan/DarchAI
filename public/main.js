function createRandomStars(numStars) {
    const container = document.querySelector(".shooting-star-container");
    
    for (let i = 0; i < numStars; i++) {
      const starContainer = document.createElement("div");
      starContainer.className = "star-container";
      starContainer.style.top = Math.random() * window.innerHeight + "px";
      starContainer.style.left = Math.random() * window.innerWidth + "px";
      
      let animationDuration = 2 + Math.random() * 5;
      starContainer.style.animationDuration = animationDuration + "s";
        
      const star = document.createElement("div");
      star.className = "star";
      star.style.animationDuration = starContainer.style.animationDuration = animationDuration + "s";
      starContainer.appendChild(star);
  
      const trail = document.createElement("div");
      trail.className = "trail";
      trail.style.animationDuration = animationDuration + "s";
      starContainer.appendChild(trail);
      
      container.appendChild(starContainer);
      setTimeout(() => {
        starContainer.remove();
        createRandomStars(1);
      }, animationDuration * 1000);
    }
  }
  
  createRandomStars(100); // You can change the number of stars to generate
