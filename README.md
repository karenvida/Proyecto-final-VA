<div align="center">
  <img src="02IMT-Positivo.png" width="70%">
</div>

# Final Project Artificial Vision

Chess is a game played by amateurs all over the world. In this way certain opening movements and checkmates were developed in order to improve the way players start a game. Traditional practice don’t give an instant feedback of the movements. In this way, the virtualization of the chessboard can help players to improve since openings and plays specify the movement of the pieces and the player can memorize them by getting feedback of their movements.

<div align="center">
  <a href="#Section 1"><b>Section 1</b></a> |
  <a href="#Section 2"><b>Section 2</b></a> |
  <a href="#Section 3"><b>Section 3</b></a> |

</div>

## Pipeline
The following pipeline was presented for recognizing the chess board:
Converting the image to gray-scale.
1. Blurring the image.
2. Getting edges by Canny.
3. Using Hough Lines to determine the principal lines of the board.
<div align="center">
  <img src="Image1.jpeg" width="50%">
</div>
5. Getting the intersection points of the found lines to get the corners of the board.
6. Clustering of the nearby points in the intersections.
7. Getting the outer corners.
<div align="center">
  <img src="Image2.jpeg" width="50%">
</div>
9. Using homography to present the perpendicular vision of the warped board.
<div align="center">
  <img src="Image3.jpeg" width="50%">
</div>


## Results

## How to run

<div align="center">
  <img src="wheel.png" width="40%">
</div># Final Project Artificial Vision
about
