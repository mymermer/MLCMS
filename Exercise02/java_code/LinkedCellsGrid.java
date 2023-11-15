package org.vadere.util.geometry;

import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.List;

import org.jetbrains.annotations.NotNull;
import org.vadere.util.geometry.shapes.VPoint;
import org.vadere.util.geometry.shapes.VRectangle;

import org.locationtech.jts.index.quadtree.Quadtree;
import org.locationtech.jts.geom.Envelope;


/**
 * A grid augmenting the position of generic objects, for faster access. O(1)
 * instead of O(n) for one fixed radius check. See
 * {@link LinkedCellsGrid#getObjects(VPoint, double)}.
 *
 *
 */
public class LinkedCellsGrid<T extends PointPositioned> implements Iterable<T> {
	final private double left;
	final private double top;
	final private double width;
	final private double height;
	private GridCell<T>[][] grid;
	private int[] gridSize = new int[2];
	private double[] cellSize = new double[2];
	private int size;
	private Quadtree quadTree;

	/**
	 * One cell in the grid. It triangleContains a mapping from points to lists of
	 * objects. This means that one can store multiple objects in one cell.
	 *
	 *
	 * @param <E>
	 *        type of objects stored in this cell.
	 */
	private class GridCell<E extends PointPositioned> {
		public List<E> objects = new ArrayList<>();

		@Override
		public int hashCode() {
			final int prime = 31;
			int result = 1;
			result = prime * result + getOuterType().hashCode();
			result = prime * result
					+ ((objects == null) ? 0 : objects.hashCode());
			return result;
		}

		@Override
		public boolean equals(Object obj) {
			if (this == obj) {
				return true;
			}
			if (obj == null) {
				return false;
			}
			if (!(obj instanceof GridCell)) {
				return false;
			}
			GridCell other = (GridCell) obj;
			if (objects == null) {
				if (other.objects != null) {
					return false;
				}
			} else if (!objects.equals(other.objects)) {
				return false;
			}
			return true;
		}

		private LinkedCellsGrid getOuterType() {
			return LinkedCellsGrid.this;
		}
	}

	private class ContainerisedElement {
		final private int[] cell;
		final private T object;

		ContainerisedElement(int[] cell, T object){
			this.cell = cell;
			this.object = object;
		}

		public int[] getCell() {
			return cell;
		}

		public T getObject() {
			return object;
		}

		@Override
		public String toString() {
			return "ContainerisedElement{" +
					"cell=" + Arrays.toString(cell) +
					", object=" + object +
					", pos= " + ((PointPositioned)object).getPosition() +
					'}';
		}
	}

	/**
	 * Generates an empty grid of GridCell&lt;T&gt; objects.
	 *
	 * @param s
	 *        grid side length
	 * @return A two-dimensional grid of size "s^2" containing GridCell&lt;T&gt;
	 *         objects.
	 */
	@SuppressWarnings("unchecked")
	private GridCell<T>[][] generateGrid(int... s) {
		// Use Array native method to create array of a type only known at run
		// time
		this.grid = (GridCell<T>[][]) Array.newInstance(GridCell.class, s);
		for (int r = 0; r < grid.length; r++) {
			// TODO [priority=medium] [task=test] changed this [20.08.2014] here 1 to r - pls check this
			for (int c = 0; c < grid[r].length; c++) {
				grid[r][c] = new GridCell<>();
			}
		}

		return this.grid;
	}

	public int getGridWidth() {
		return gridSize[0];
	}

	public int getGridHeight() {
		return gridSize[1];
	}

	public LinkedCellsGrid(@NotNull final VRectangle bound, double sideLength) {
		this(bound.x, bound.y, bound.width, bound.height, sideLength);
	}

	/**
	 * Generates a LinkedCellsGrid with given dimension, position and number of
	 * items on one side.
	 *
	 * @param left
	 *        x-position of top left corner
	 * @param top
	 *        y-position of top left corner
	 * @param width
	 *        width of the grid, in world units (e.g. [m])
	 * @param height
	 *        height of the grid, in world units (e.g. [m])
	 * @param sideLength
	 *        number of items on one side. the total number of grid cells
	 *        equals sideLength^2
	 */
	public LinkedCellsGrid(double left, double top, double width,
						   double height, double sideLength) {
		this.left = left;
		this.top = top;
		this.width = width;
		this.height = height;
		this.size = 0;
		this.quadTree = new Quadtree(); //quad tree deifned here


		// create grid
		/*
		 * this.gridSize = (int) Math.bound(1,
		 * Math.ceil(Math.bound(width, height) / sideLength));
		 * this.cellSize = Math.bound(width, height) / gridSize;
		 * this.grid = generateGrid(gridSize, gridSize);
		 */

		this.gridSize[0] = (int) Math.max(1, this.width / sideLength);
		this.gridSize[1] = (int) Math.max(1, this.height / sideLength);
		this.cellSize[0] = this.width / gridSize[0];
		this.cellSize[1] = this.height / gridSize[1];

		this.grid = generateGrid(this.gridSize[0], this.gridSize[1]);
	}

	/**
	 * Determines the discrete grid position (x,y) given a point with double
	 * coordinates.
	 *
	 * @param pos
	 *        a given position with coordinate values of type double
	 * @return the position in the grid, from 0 to this.gridSize-1 in both
	 *         coordinates.
	 */
	public int[] gridPos(VPoint pos) {
		// compute position in the grid
		int iX = (int) Math.max(
				0,
				Math.min(this.gridSize[0] - 1,
						Math.floor((pos.x - left) / width * this.gridSize[0])));
		int iY = (int) Math.max(
				0,
				Math.min(this.gridSize[1] - 1,
						Math.floor((pos.y - top) / height * this.gridSize[1])));

		return new int[] {iX, iY};
	}

	public int[][] getCellObjectCount(){
		int[][] count = new int[this.gridSize[0]][this.gridSize[1]];
		for (int r = 0; r < grid.length; r++) {
			for (int c = 0; c < grid[r].length; c++) {
				count[r][c] = grid[r][c].objects.size();
			}
		}
		return count;
	}


	/**
	 * Adds the given object to the grid and the QuadTree.
	 * @param object the object to be added
	 */
	public synchronized void addObject(final T object) {
		int[] gridPos = gridPos(object.getPosition());
		grid[gridPos[0]][gridPos[1]].objects.add(object);

		// Create an Envelope representing the spatial extent of the object
		Envelope envelope = new Envelope(
				object.getPosition().getX(),
				object.getPosition().getX(),
				object.getPosition().getY(),
				object.getPosition().getY()
		);



		// Add the object to the QuadTree
		quadTree.insert( envelope, object);


		size++;
	}

	public void moveObject(final T object, final VPoint oldPosition) {
		removeObject(object, oldPosition);
		addObject(object);
	}



	/**
	 * Returns a list of objects within a certain radius of a given position using quadTree.
	 * @param pos The center position to search around.
	 * @param radius The radius to search within.
	 * @return A list of objects within the search radius.
	 */
	public synchronized List<T> getObjects(final VPoint pos, final double radius) {
		final List<T> result = new LinkedList<>();

		// Create an Envelope representing the search area
		Envelope searchEnvelope = new Envelope(
				pos.getX() - radius,
				pos.getX() + radius,
				pos.getY() - radius,
				pos.getY() + radius
		);

		// Query the Quadtree for objects within the search envelope
		List<Object> potentialNeighbors = new ArrayList<>(); // Explicitly specifying type parameter
		quadTree.query(searchEnvelope, potentialNeighbors::add);

		// Filter potential neighbors based on the actual distance
		for (Object obj : potentialNeighbors) {
			if (obj instanceof PointPositioned) {
				PointPositioned pointPositioned = (PointPositioned) obj;
				if (pointPositioned.getPosition().distance(pos) < radius) {
					result.add((T) obj);
				}


			}
		}
		return result;
	}






	/**
	 * Removes the specified object from the grid and quadtree.
	 * 
	 * @param object the object to be removed
	 */
	public synchronized void removeObject(T object) {

		// Create an Envelope representing the spatial extent of the object
		Envelope envelope = new Envelope(
				object.getPosition().getX(),
				object.getPosition().getX(),
				object.getPosition().getY(),
				object.getPosition().getY()
		);

		quadTree.remove(envelope, object);




		int[] gridPos = gridPos(object.getPosition());
		if(grid[gridPos[0]][gridPos[1]].objects.removeIf(element -> element.equals(object))){
			size--;
		}

	}

	/**
	 * Removes the given object from the grid and quadtree at the given old position.
	 * 
	 * @param object the object to be removed
	 * @param oldPosition the old position of the object
	 */
	public synchronized void removeObject(T object, final VPoint oldPosition) {

		// Create an Envelope representing the spatial extent of the object
		Envelope envelope = new Envelope(
				oldPosition.getX(),
				oldPosition.getX(),
				oldPosition.getY(),
				oldPosition.getY()
		);

		quadTree.remove(envelope, object);

		int[] gridPos = gridPos(oldPosition);
		if(grid[gridPos[0]][gridPos[1]].objects.removeIf(element -> element.equals(object))){
			size--;
		}
	}

	/**
	 * Removes all objects from the grid and quadTree.
	 */
	public void clear() {
		grid = generateGrid(gridSize[0], gridSize[1]);
		size = 0;
		quadTree= new Quadtree();
	}

	public List<T> getElements() {
		List<T> elements = new ArrayList<>();
		for (int r = 0; r < grid.length; r++) {
			// TODO [priority=medium] [task=test] changed this [20.08.2014] here 1 to r - pls check this
			for (int c = 0; c < grid[r].length; c++) {
				elements.addAll(grid[r][c].objects);
			}
		}

		return elements;
	}

	/**
	 * Provides an iterator over the objects stored in the grid.
	 */
	@Override
	public Iterator<T> iterator() {
		return getElements().iterator();
	}

	/**
	 * Returns the size (number of different keys &lt;T&gt;) of List.
	 *
	 * @return the size (number of different keys &lt;T&gt;) of List
	 */
	public int size() {
		return size;
	}

	/**
	 * Tests whether the linked cells grid triangleContains an object that equals(the
	 * given object). The complexity of this operation is O(N), N = number of
	 * objects in the grid.
	 *
	 * @param element
	 * @return
	 */
	public boolean contains(final T element) {
		for (Iterator<T> iter = this.iterator(); iter.hasNext();) {
			if (iter.next().equals(element)) {
				return true;
			}
		}
		return false;
	}

	public List<ContainerisedElement> getElementContainer(final T element){
		List<ContainerisedElement> elements  = new ArrayList<>();
		for (int r = 0; r < grid.length; r++) {
			// TODO [priority=medium] [task=test] changed this [20.08.2014] here 1 to r - pls check this
			for (int c = 0; c < grid[r].length; c++) {
				if (grid[r][c].objects.contains(element)){
					elements.add(new ContainerisedElement(new int[]{r, c}, element));
				}
			}
		}
		return elements;
	}

	@Override
	public int hashCode() {
		final int prime = 31;
		int result = 1;
		long temp;
		temp = java.lang.Double.doubleToLongBits(cellSize[0]);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		result = prime * result + Arrays.hashCode(grid);
		result = prime * result + gridSize[1];
		temp = java.lang.Double.doubleToLongBits(height);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = java.lang.Double.doubleToLongBits(left);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = java.lang.Double.doubleToLongBits(top);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		temp = java.lang.Double.doubleToLongBits(width);
		result = prime * result + (int) (temp ^ (temp >>> 32));
		return result;
	}

	@Override
	public boolean equals(Object obj) {
		if (this == obj) {
			return true;
		}
		if (obj == null) {
			return false;
		}
		if (!(obj instanceof LinkedCellsGrid)) {
			return false;
		}
		LinkedCellsGrid other = (LinkedCellsGrid) obj;
		if (java.lang.Double.doubleToLongBits(cellSize[0]) != java.lang.Double
				.doubleToLongBits(other.cellSize[0])) {
			return false;
		}
		if (!Arrays.deepEquals(grid, other.grid)) {
			return false;
		}
		if (!Arrays.equals(gridSize, other.gridSize)) {
			return false;
		}
		if (java.lang.Double.doubleToLongBits(height) != java.lang.Double
				.doubleToLongBits(other.height)) {
			return false;
		}
		if (java.lang.Double.doubleToLongBits(left) != java.lang.Double
				.doubleToLongBits(other.left)) {
			return false;
		}
		if (java.lang.Double.doubleToLongBits(top) != java.lang.Double
				.doubleToLongBits(other.top)) {
			return false;
		}
		if (java.lang.Double.doubleToLongBits(width) != java.lang.Double
				.doubleToLongBits(other.width)) {
			return false;
		}
		return true;
	}

}
