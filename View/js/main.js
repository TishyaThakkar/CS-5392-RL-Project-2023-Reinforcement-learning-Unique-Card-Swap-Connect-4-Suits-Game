

// Dots and Boxes Game Class
class DotsAndBoxes {

    // Private Variables 
    #w;
    #h;
    #score;
    #scoreBoard;
    #board;
    #players;
    #turn;
    #stage;
    #actionMap;
    #winningMap;
    #colors = {background: "#FFFFFA", circle: "#F06449", lines: "#227C9D", texts: "#230903"};

    constructor(width , height, canvasId) {        
        // width and height of the board
        this.#w = width ;
        this.#h = height;
        this.#actionMap = {}
        this.#winningMap = {};
        this.#score = {0:0 , 1:0};
        this.#scoreBoard = this.#getScoreBoard(this.#w,this.#h);
        this.#board = this.#getEmptyBoard(this.#w,this.#h);
        this.#players = {0: true, 1: true};
        this.#turn = 0;
        this.#getNewStage();
        this.#prepareCanvas(this.#stage, this.#board);
        this.displayAll();
    }

    #getScoreBoard(w, h) {
        const twoDimArray = [];
        for(let i=0;i<h;i++) {
            let subarray = [];
            for(let j=0;j<w;j++) 
                subarray.push(-1);
            twoDimArray.push(subarray)
        }
        return twoDimArray;
    }

    #getEmptyBoard(w, h) {
        w = ( w *2 ) + 1;
        h = ( h * 2 ) + 1;

        const twoDimArray = [];
        var action = 0
        var winBlock = 0;
        
        for(let i=0;i<h;i++) {
            let subarray = [];
            if(i%2==0) {
                for(let j=0;j<w;j++) {
                    subarray.push(j%2==0 ? -1 : 0)
                    if(subarray[subarray.length-1] == 0) {
                        this.#actionMap[action] = {x:i,y:j};
                        //this.#actionMap[i+" "+j] = action;
                        action = action+1;
                    }
                }
            } else {
                for(let j=0;j<w;j++) { 
                    subarray.push(j%2==0 ? 0 : -1)
                    if(subarray[subarray.length-1] == 0) {
                        this.#actionMap[action] = {x:i,y:j};
                        //this.#actionMap[i+" "+j] = action;
                        action = action+1;
                    } else if(subarray[subarray.length-1] == -1) {
                        this.#winningMap[i+" "+j] = winBlock;
                        this.#winningMap[winBlock] = {x:i,y:j};
                        winBlock = winBlock + 1;
                    }
                }
            }
            twoDimArray.push(subarray)
        }
        return twoDimArray;
    }

    #prepareCanvas(stage, board) {
        stage.clear();
        var myText = "\nScore : { Player1 = "+ this.#score[0]  +" ,    Player2 = "+this.#score[1] + "}"
                        +"                         Turn: Player"+  Number(this.#turn+1) ;
        let hw = new createjs.Text(myText, "1.1em sans-serif", this.#colors.texts);
        stage.addChild(hw);
        hw.x = 100;
        hw.y = 10;
        for(let i=0;i<board.length; i++) {
            for(let j=0;j<board[i].length; j++) {
                if(board[i][j] == -1 && j%2 == 0) {
                    var circle = new createjs.Shape();
                    circle.graphics
                        .beginFill(this.#colors.circle)
                        .drawCircle( 100 + (50*j), 100 + (50 * i) , 10);
                    stage.addChild(circle);
                } else if(board[i][j] == 0 && j%2 == 0) {
                    var line = new createjs.Shape();
                    line.graphics.setStrokeStyle(35).beginStroke(this.#colors.background);
                    line.graphics.moveTo(100 + (50*(j)), 115 + (50 * (i-1)));
                    line.graphics.lineTo(100 + (50*(j)), 85 + (50 * (i+1)));
                    line.graphics.endStroke();
                    line.addEventListener("click", (event) => { this.playWithCoOrdinates( i , j)})
                    stage.addChild(line);
                } else if(board[i][j] == 0 && j%2 == 1) {
                    var line = new createjs.Shape();
                    line.graphics.setStrokeStyle(35).beginStroke(this.#colors.background);
                    line.graphics.moveTo(115 + (50*(j-1)), 100 + (50 * (i)));
                    line.graphics.lineTo(85 + (50*(j+1)), 100 + (50 * (i)));
                    line.graphics.endStroke();
                    line.addEventListener("click", (event) => { this.playWithCoOrdinates( i , j)})
                    stage.addChild(line);
                } else if(board[i][j] == 1 && j%2 == 0) {
                    var line = new createjs.Shape();
                    line.graphics.setStrokeStyle(15).beginStroke(this.#colors.lines);
                    line.graphics.moveTo(100 + (50*(j)), 115 + (50 * (i-1)));
                    line.graphics.lineTo(100 + (50*(j)), 85 + (50 * (i+1)));
                    line.graphics.endStroke();
                    stage.addChild(line);
                } else if(board[i][j] == 1 && j%2 == 1) {
                    var line = new createjs.Shape();
                    line.graphics.setStrokeStyle(15).beginStroke(this.#colors.lines);
                    line.graphics.moveTo(115 + (50*(j-1)), 100 + (50 * (i)));
                    line.graphics.lineTo(85 + (50*(j+1)), 100 + (50 * (i)));
                    line.graphics.endStroke();
                    stage.addChild(line);
                }

            }
        }
        var c = 0;
        for(var i=0;i<this.#scoreBoard.length;i++) {
            for(var j=0;j<this.#scoreBoard[i].length;j++) {
                if(this.#scoreBoard[i][j] != -1) {
                    var cor = this.#winningMap[c];    
                    let hw = new createjs.Text(Number(this.#scoreBoard[i][j]+1), "bold 2em sans-serif", this.#colors.texts);
                    stage.addChild(hw);
                    hw.x = 90 + (cor.y * 50);
                    hw.y = 90 + (cor.x * 50);
                }
                c = c + 1;
            }
        }

        stage.update();
    }

    #getNewStage() {
        this.#stage = new createjs.Stage("mycanvas");
    }

    #deleteOldStage() {
        this.#stage.enableMouseOver(-1);
        this.#stage.enableDOMEvents(false);
        this.#stage.removeAllEventListeners();
        this.#stage.removeAllChildren();
        this.#stage.canvas = null;
        this.#stage = null;
    }

    // Incremental Turn 
    playWithCoOrdinates(x,y) {
        console.log(x,y);
        this.#deleteOldStage()
        this.#getNewStage();
        this.#board[x][y] = 1;
        var checkScoreResults = this.#checkForScore(this.#board, x, y);
        var shouldKeepTurn =  checkScoreResults[0];
        var scoreEarned = checkScoreResults[1];
        if(!shouldKeepTurn) 
            this.#turn = (this.#turn+1)%2 
        this.#prepareCanvas(this.#stage, this.#board)
        console.log("------------------------------------");
        this.displayAll();
        console.log("------------------------------------");
    }

    #checkForScore(board, x, y) {
        var hasScored = false;
        var scored = 0;
        var turn = this.#turn;
        var width = this.#board[0].length;
        var height = this.#board.length;

        if(x%2 == 0) { // check Squares up and down
            // square above ?
            if ( x - 2 >= 0 && y-1 >= 0 && y + 1 < width ) {
                if(board[x-2][y] == 1 && board[x-1][y-1] == 1 && board[x-1][y+1] == 1 ) {
                    hasScored = true;
                    //center
                    var num = this.#winningMap[(x-1)+" "+(y)]
                    console.log((x-1), (y), num, 'above');
                    var sx = Math.floor(num / this.#w);
                    var sy = num % this.#w;
                    console.log((x+1), (y), num, 'below', sx, sy);
                    this.#scoreBoard[sx][sy] = turn;
                    this.#score[turn] = this.#score[turn] + 1;
                    scored = scored + 1;
                    console.log('Score!!!');
                }
            }
            // square below ?
            if ( x + 2 < height && y-1 >= 0 && y + 1 < width ) {
                if(board[x+2][y] == 1 && board[x+1][y-1] == 1 && board[x+1][y+1] == 1 ) {
                    hasScored = true;
                    //center
                    var num = this.#winningMap[(x+1)+" "+(y)]
                    console.log((x+1), (y), num, 'below');
                    var sx =  Math.floor(num / this.#w);
                    var sy = num % this.#w;
                    console.log((x+1), (y), num, 'below', sx, sy);
                    this.#scoreBoard[sx][sy] = turn;
                    this.#score[turn] = this.#score[turn] + 1;
                    scored = scored + 1;
                    console.log('Score!!!');
                }
            }
            
        } else { // check Squares right and left
            // square left ?
            if ( y - 2 >= 0 && x-1 >= 0 && x + 1 < width ) {
                if(board[x][y-2] == 1 && board[x-1][y-1] == 1 && board[x+1][y-1] == 1 ) {
                    hasScored = true;
                    //center
                    var num = this.#winningMap[(x)+" "+(y-1)]
                    var sx = Math.floor(num / this.#w);
                    var sy = num % this.#w;
                    console.log((x), (y-1), num, 'left', sx, sy);
                    this.#scoreBoard[sx][sy] = turn;
                    this.#score[turn] = this.#score[turn] + 1;
                    scored = scored + 1;
                    console.log('Score!!!');
                }
            }
            // square right ?
            if ( y + 2 < width && x-1 >= 0 && x + 1 < width ) {
                if(board[x][y+2] == 1 && board[x-1][y+1] == 1 && board[x+1][y+1] == 1 ) {
                    hasScored = true;
                    //center
                    var num = this.#winningMap[(x)+" "+(y+1)]
                    var sx = Math.floor(num / this.#w);
                    var sy = num % this.#w;
                    console.log((x), (y+1), num, 'right', sx, sy);
                    this.#scoreBoard[sx][sy] = turn;
                    this.#score[turn] = this.#score[turn] + 1;
                    scored = scored + 1;
                    console.log('Score!!!');
                }
            }
        }

        return [hasScored, scored];
    }

    playWithAction(action) {
        var x = this.#actionMap[action].x;
        var y = this.#actionMap[action].y;
        
        this.playWithCoOrdinates(x,y);
    }

    displayAll() {
        console.log(this.#w)
        console.log(this.#h)
        console.log(this.#score)
        console.log(this.#scoreBoard)
        console.log(this.#board)
        console.log(this.#players)
        console.log(this.#turn)
        console.log(this.#actionMap)
        console.log(this.#winningMap)

    }


}

var game;
function init() {
    game = new DotsAndBoxes(5,4, "mycanvas");
}

