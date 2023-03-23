$( document ).ready(function() {
    
    var board = [11,12,13,14,21,22,23,24,31,32,33,34,41,42,43,44];
    var suites = [-1,-1];
    var selectedCards = [];
    var aiselectedCards = [];


    $("#bNewGame").on('click',function (){
        board = shuffleArray(board);
        displayBoard();
    });

    $(".img-container").on('click', function(e) {
        //console.log(e);
        resetStyleForAllImages();
        //console.log(selectedCards);
        if(selectedCards.length>1) {
            selectedCards.shift();
        }
        var idStr = $(e.target).attr('id')
        if(idStr.charAt(idStr.length-1) == 'i')
            var target = Number(idStr.substring(1, idStr.length-1))
        else 
        var target = Number(idStr.substring(1, idStr.length))
        selectedCards.push(target);
        //console.log(selectedCards);
        //console.log(idStr,target);
        highlightSelectedImages();
        
    });

    function shuffleArray(array) { 
        //return array.sort( ()=>Math.random()-0.5 );
        let currentIndex = array.length,  randomIndex;

        while (currentIndex != 0) {
          randomIndex = Math.floor(Math.random() * currentIndex);
          currentIndex--;
          [array[currentIndex], array[randomIndex]] = [
            array[randomIndex], array[currentIndex]];
        }
      
        return array;
    } 

    function displayBoard() {
        var i = 0;
        var size = board.length;
        while(i<size) {
            $("#p"+i+"i").attr("src",getImage(board[i]));
            i++;
        }
    }

    function getImage(i) {
        return "img/"+i+".png";
    }

    function resetStyleForAllImages() {
        var i = 0;
        var size = board.length;
        while(i<size) {
            $("#p"+i).attr("style","");
            i++;
        }
    }

    function highlightSelectedImages() {
        var i = 0, len = selectedCards.length;
        while (i < len) {
            $("#p"+selectedCards[i]).attr("style","border: 5px solid #ef476f");
            //06d6a0, 118ab2,  ffd166
            i++
        }

    }


});