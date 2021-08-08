/*

Data Cleaning 

*/

SELECT * FROM housing.dbo.HousingData;


-- Fix Date Format

SELECT New_SaleDate,CONVERT(DATE,SaleDate)
FROM housing.dbo.HousingData;



ALTER TABLE HousingData
ADD New_SaleDate DATE;

UPDATE HousingData
SET New_SaleDate = CONVERT(DATE,SaleDate);

---------------------------------------------------------

-- fill null values in PropertyAddress Column

SELECT a.ParcelID,a.PropertyAddress,b.ParcelID,b.PropertyAddress
FROM HousingData a JOIN HousingData b
ON a.ParcelID=b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress is NULL;

UPDATE a
SET PropertyAddress= ISNULL(a.PropertyAddress,b.PropertyAddress)
FROM HousingData a JOIN HousingData b
ON a.ParcelID=b.ParcelID
AND a.[UniqueID ] <> b.[UniqueID ]
WHERE a.PropertyAddress is NULL;

---------------------------------------------------------------
-- Devide property address into (Address,city)

SELECT * FROM housing.dbo.HousingData;


SELECT Substring(PropertyAddress,1,CHARINDEX(',',PropertyAddress) -1) AS Address,
SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,LEN(PropertyAddress)) AS Address
From housing.dbo.HousingData;


ALTER TABLE housing.dbo.HousingData
ADD Address NVARCHAR(255);


UPDATE housing.dbo.HousingData
SET Address= Substring(PropertyAddress,1,CHARINDEX(',',PropertyAddress) -1);

ALTER TABLE housing.dbo.HousingData
ADD city NVARCHAR(255);

UPDATE housing.dbo.HousingData
SET city = SUBSTRING(PropertyAddress,CHARINDEX(',',PropertyAddress)+1,LEN(PropertyAddress));

------------------------------------------------------

---- Devide Owner address into (Address,city,state)

SELECT * FROM housing.dbo.HousingData;


SELECT 
PARSENAME(REPLACE(OwnerAddress, ',' , '.'),3) AS Address,
PARSENAME(REPLACE(OwnerAddress, ',' , '.'),2)AS City,
PARSENAME(REPLACE(OwnerAddress, ',' , '.'),1) AS State
FROM housing.dbo.HousingData;

ALTER TABLE housing.dbo.HousingData
ADD Owner_Address NVARCHAR(255),Owner_City NVARCHAR(255),Owner_State NVARCHAR(255);

UPDATE housing.dbo.HousingData
SET Owner_Address= PARSENAME(REPLACE(OwnerAddress, ',' , '.'),3);


UPDATE housing.dbo.HousingData
SET Owner_City=PARSENAME(REPLACE(OwnerAddress, ',' , '.'),2);


UPDATE housing.dbo.HousingData
SET Owner_State= PARSENAME(REPLACE(OwnerAddress, ',' , '.'),1);

----------------------------------------------------------------------

-- Change Y and N to Yes and No in (SoldASVacant) Column

SELECT DISTINCT(SoldAsVacant),Count(SoldAsVacant)
FROM housing.dbo.HousingData
GROUP BY SoldAsVacant
ORDER BY 2;

-- So we have 52 rows with Y & 399 rows with N

SELECT SoldAsVacant,
CASE WHEN SoldAsVacant='Y' THEN 'Yes'
     WHEN SoldAsVacant='N' THEN 'NO'
	 ELSE SoldAsVacant
	 END
FROM housing.dbo.HousingData;

UPDATE housing.dbo.HousingData
SET SoldAsVacant= CASE WHEN SoldAsVacant='Y' THEN 'Yes'
     WHEN SoldAsVacant='N' THEN 'NO'
	 ELSE SoldAsVacant
	 END;


------------------------------------------------------------------

-- remove duplicate rows

WITH RowNum AS(
Select *,
	ROW_NUMBER() OVER (
	PARTITION BY ParcelID,
				 PropertyAddress,
				 SalePrice,
				 SaleDate,
				 LegalReference
				 ORDER BY
					UniqueID
					) row_num
From housing.dbo.HousingData
)
DELETE FROM RowNum
WHERE row_num>1;

---------------------------------------------------------------

-- Delete unused Columns

SELECT * FROM housing.dbo.HousingData;

ALTER TABLE housing.dbo.HousingData
DROP COLUMN OwnerAddress,PropertyAddress,SaleDate;



-------------------------------------------------------------


