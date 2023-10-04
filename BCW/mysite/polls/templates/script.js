document.addEventListener("DOMContentLoaded", function() {
  var body = document.querySelector("body");
  var classifyButton = document.querySelector("input[type='submit']");

  classifyButton.addEventListener("mouseover", function() {
    body.style.backgroundImage = "url('hover-background.jpg')";
  });

  classifyButton.addEventListener("mouseout", function() {
    body.style.backgroundImage = "url('default-background.jpg')";
  });
});
