window.addEventListener('load', function(){

	var canvas = document.querySelector("#canvas");
	var context = canvas.getContext("2d");
	canvas.width = 280;
	canvas.height = 280;

	var Mouse = {x:0, y:0};
	var lastMouse = {x:0, y:0};
	context.fillStyle = "white";
	context.fillRect(0, 0, canvas.width, canvas.height);
	context.color = "black";
	context.lineWidth = 15;
    context.lineJoin = context.lineCap = 'round';

	debug();
	var isIdle = true;
  function drawstart(event) {
    context.beginPath();
    context.moveTo(event.pageX - canvas.offsetLeft, event.pageY - canvas.offsetTop);
    context.color = "black";
	context.lineWidth = 15;
    context.lineJoin = context.lineCap = 'round';

    isIdle = false;
  }
  function drawmove(event) {
    if (isIdle) return;
    context.lineTo(event.pageX - canvas.offsetLeft, event.pageY - canvas.offsetTop);
    context.stroke();
  }
  function drawend(event) {
    if (isIdle) return;
    drawmove(event);
    isIdle = true;
  }
  function touchstart(event) { drawstart(event.touches[0]) }
  function touchmove(event) { drawmove(event.touches[0]); event.preventDefault(); }
  function touchend(event) { drawend(event.changedTouches[0]) }

  canvas.addEventListener('touchstart', touchstart, false);
  canvas.addEventListener('touchmove', touchmove, false);
  canvas.addEventListener('touchend', touchend, false);        

  canvas.addEventListener('mousedown', drawstart, false);
  canvas.addEventListener('mousemove', drawmove, false);
  canvas.addEventListener('mouseup', drawend, false);

  
  function debug() {
	$("#clearButton").on("click", function() {
		context.clearRect( 0, 0, 280, 280 );
		context.fillStyle="white";
		context.fillRect(0,0,canvas.width,canvas.height);
	});
}
}, false);


	


